"""Rotinas do laboratório CNNCrack: dados, CNN MATLAB, treino e exportação.

Referência: Shihang Feng, CNN_lab.zip (2018), disponibilizado pelo UTAM.
Rotinas CNN originais: Rasmus Berg Palm (2012); licença em LICENSE-CNN_lab.
O notebook concentra o protocolo e as figuras; este módulo não executa ao importar.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import platform
from time import perf_counter
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch import nn
from torch.nn import functional as F


LAB_URL = "http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.CNN/Chapter.CNNCrack/lab.html"
ZIP_URL = LAB_URL.rsplit("/", 1)[0] + "/CNN_lab.zip"
ZIP_SHA256 = "fd4279b568b1b3261d3c999e884fa70525608acf792815c5388fb4c19fdb6595"
BINARY_CLASSES = {0: "Fundo", 1: "Fratura"}
MULTICLASS_CLASSES = {0: "Fundo", 1: "Fratura grande", 2: "Fratura pequena"}


def load_lab_data(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Lê os dois MAT em memória, verificando o SHA-256 do ZIP local/oficial."""
    cache_dir = Path(cache_dir)
    path = cache_dir / "CNN_lab.zip"
    payload = path.read_bytes() if path.exists() else b""
    if sha256(payload).hexdigest() != ZIP_SHA256:
        try:
            request = Request(ZIP_URL, headers={"User-Agent": "exercise04/1.0"})
            with urlopen(request, timeout=60) as response:
                payload = response.read()
        except OSError as error:
            raise RuntimeError(
                f"Não foi possível baixar {ZIP_URL}. Baixe o ZIP pelo navegador e "
                f"salve-o em {path.resolve()}; em seguida, execute novamente."
            ) from error
        if sha256(payload).hexdigest() != ZIP_SHA256:
            raise ValueError("O ZIP recebido difere do pacote oficial verificado neste laboratório.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".zip.part")
        temporary.write_bytes(payload)
        temporary.replace(path)

    with ZipFile(BytesIO(payload)) as archive:
        image_mat = loadmat(BytesIO(archive.read("CNN_lab/data/image_input.mat")))
        train_mat = loadmat(
            BytesIO(archive.read("CNN_lab/data/train_new.mat")), simplify_cells=True
        )["train"]
    image = np.asarray(image_mat["image_input"], dtype=np.float32)
    if image.shape != (457, 977) or not np.isfinite(image).all():
        raise ValueError("Imagem oficial inválida; esperada uma matriz finita 457 × 977.")
    # MATLAB: primeira saída = fratura; segunda = fundo. Coordenadas começam em 1.
    annotations = pd.DataFrame({
        "x": np.asarray(train_mat["trainx"], dtype=int).ravel() - 1,
        "z": np.asarray(train_mat["trainz"], dtype=int).ravel() - 1,
        "label": 1 - np.argmax(train_mat["label"], axis=0),
    })
    annotations = validate_annotations(annotations, image.shape, BINARY_CLASSES)
    return image, annotations


def validate_annotations(
    table: pd.DataFrame,
    image_shape: tuple[int, int],
    classes: dict[int, str],
    counts: dict[int, int] | None = None,
    allow_duplicates: bool = True,
) -> pd.DataFrame:
    """Verifica índices inteiros, limites e classes; preserva repetições oficiais."""
    if not {"x", "z", "label"}.issubset(table.columns) or len(table) == 0:
        raise ValueError("O CSV deve conter linhas e as colunas x,z,label.")
    table = table[["x", "z", "label"]].copy().reset_index(drop=True)
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("Coordenadas e rótulos devem ser inteiros finitos.")
    table = table.astype(int)
    height, width = image_shape
    if not (table.x.between(0, width - 1).all() and table.z.between(0, height - 1).all()):
        raise ValueError("Coordenadas fora da imagem; x é coluna e z é linha, começando em zero.")
    if set(table.label) != set(classes):
        raise ValueError(f"São necessárias as classes {classes}; encontradas {sorted(set(table.label))}.")
    if table.groupby(["x", "z"]).label.nunique().max() > 1:
        raise ValueError("Um mesmo pixel recebeu classes diferentes.")
    if not allow_duplicates and table.duplicated(["x", "z"]).any():
        raise ValueError("Escolha pixels distintos nas anotações manuais.")
    if counts is not None:
        actual = table.label.value_counts().to_dict()
        if actual != counts:
            raise ValueError(f"Contagem esperada: {counts}; encontrada: {actual}.")
    return table


def read_annotations(
    path: Path, image_shape: tuple[int, int], classes: dict[int, str], counts: dict[int, int]
) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Salve {path} no anotador da célula anterior e execute esta célula novamente."
        )
    return validate_annotations(
        pd.read_csv(path), image_shape, classes, counts, allow_duplicates=False
    )


def build_annotation_datasets(
    big: pd.DataFrame, small: pd.DataFrame, image_shape: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deriva dois problemas dos mesmos 800 registros, sem escrever nas fontes.

    Cada fonte contém 200 pontos de fundo e 200 de fratura. A concatenação
    preserva todas as linhas, inclusive eventuais repetições entre as fontes.
    No problema multiclasse, apenas a fratura de small muda de label=1 para 2.
    source e source_row permitem rastrear cada registro até a anotação original.
    """
    sources = []
    for name, table in (("big", big), ("small", small)):
        source = validate_annotations(
            table, image_shape, BINARY_CLASSES, {0: 200, 1: 200}, allow_duplicates=False
        )
        source["source"] = name
        source["source_row"] = np.arange(1, len(source) + 1)
        sources.append(source)
    binary = pd.concat(sources, ignore_index=True)
    multiclass = binary.copy(deep=True)
    multiclass.loc[(multiclass.source == "small") & (multiclass.label == 1), "label"] = 2
    validate_annotations(binary, image_shape, BINARY_CLASSES, {0: 400, 1: 400})
    validate_annotations(
        multiclass, image_shape, MULTICLASS_CLASSES, {0: 400, 1: 200, 2: 200}
    )
    return binary, multiclass


def split_by_center(
    annotations: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Divide centros únicos por classe; duplicatas ficam no mesmo subconjunto.

    Recortes vizinhos ainda podem se sobrepor. Esta validação interna não é um
    teste independente em outra rocha/imagem.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("A fração de validação deve estar entre 0 e 1.")
    rng = np.random.default_rng(seed)
    validation_centers = set()
    for _, group in annotations.groupby("label", sort=True):
        centers = group[["x", "z"]].drop_duplicates().to_numpy()
        if len(centers) < 2:
            raise ValueError("Cada classe precisa de pelo menos dois centros distintos.")
        n_val = min(len(centers) - 1, max(1, round(len(centers) * validation_fraction)))
        for row in centers[rng.permutation(len(centers))[:n_val]]:
            validation_centers.add(tuple(row))
    mask = np.array([
        (x, z) in validation_centers for x, z in annotations[["x", "z"]].to_numpy()
    ])
    return np.flatnonzero(~mask), np.flatnonzero(mask)


class PatchExtractor:
    """Recortes 17 × 17 com borda replicada e interpolação bilinear 32 × 32.

    sliding_window_view é uma vista, sem materializar todos os patches. Apenas
    o lote solicitado é copiado. align_corners=True preserva os nós de interp2.
    """

    def __init__(self, image: np.ndarray, radius: int = 8, output_size: int = 32):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2 or not np.isfinite(image).all():
            raise ValueError("A imagem deve ser uma matriz 2D finita.")
        if radius < 1 or output_size < 1:
            raise ValueError("O raio e o tamanho de saída devem ser positivos.")
        self.shape = image.shape
        self.radius = radius
        self.output_size = output_size
        self.patch_size = 2 * radius + 1
        padded = np.pad(image, radius, mode="edge")
        self.windows = np.lib.stride_tricks.sliding_window_view(
            padded, (self.patch_size, self.patch_size)
        )

    def at(self, x: np.ndarray, z: np.ndarray) -> torch.Tensor:
        x, z = np.asarray(x), np.asarray(z)
        if x.ndim != 1 or x.shape != z.shape or x.size == 0:
            raise ValueError("Forneça vetores x e z não vazios com o mesmo tamanho.")
        if not (np.equal(x, np.rint(x)).all() and np.equal(z, np.rint(z)).all()):
            raise ValueError("As coordenadas dos patches devem ser inteiras.")
        x, z = x.astype(int), z.astype(int)
        if not ((x >= 0).all() and (x < self.shape[1]).all()
                and (z >= 0).all() and (z < self.shape[0]).all()):
            raise ValueError("Centro de patch fora da imagem.")
        patches = torch.from_numpy(np.array(self.windows[z, x], copy=True))[:, None]
        return F.interpolate(
            patches, size=(self.output_size, self.output_size), mode="bilinear",
            align_corners=True,
        )

    def from_annotations(self, annotations: pd.DataFrame) -> torch.Tensor:
        return self.at(annotations.x.to_numpy(), annotations.z.to_numpy())


class MatlabConv2d(nn.Conv2d):
    """Convolução espacial como convn, incluindo a inversão espacial do kernel."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.conv2d(values, self.weight.flip(-2, -1), self.bias, self.stride, self.padding)


class CrackCNN(nn.Module):
    """Duas convoluções válidas + sigmoide + média 2 × 2 + saída sigmoide.

    A ordem das saídas em Python é label=0,1[,2]. Cada saída é um escore
    independente, como no MATLAB; os escores não somam necessariamente 1.
    """

    def __init__(self, classes: int = 2, channels: tuple[int, int] = (6, 12), kernel_size: int = 5):
        super().__init__()
        if classes < 2 or len(channels) != 2 or min(channels) < 1 or kernel_size < 1:
            raise ValueError("Arquitetura inválida.")
        spatial = 32
        for _ in range(2):
            spatial -= kernel_size - 1
            if spatial <= 0 or spatial % 2:
                raise ValueError("Cada convolução deve produzir tamanho positivo e par antes do pooling.")
            spatial //= 2
        self.classes, self.channels, self.kernel_size = classes, tuple(channels), kernel_size
        self.features = nn.Sequential(
            MatlabConv2d(1, channels[0], kernel_size), nn.Sigmoid(), nn.AvgPool2d(2),
            MatlabConv2d(channels[0], channels[1], kernel_size), nn.Sigmoid(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Linear(channels[1] * spatial * spatial, classes)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values)
        # MATLAB concatena mapas, cada um vetorizado por colunas (ordem Fortran).
        flattened = features.transpose(-2, -1).contiguous().flatten(1)
        return torch.sigmoid(self.classifier(flattened))


def matlab_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = F.one_hot(labels, num_classes=scores.shape[1]).to(scores.dtype)
    return 0.5 * (scores - target).square().sum(dim=1).mean()


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 20
    learning_rate: float = 1.0
    channels: tuple[int, int] = (6, 12)
    kernel_size: int = 5
    seed: int = 0
    weight_decay: float = 0.0
    patience: int | None = None
    min_delta: float = 1e-5
    restore_best: bool = False


@dataclass
class TrainResult:
    model: CrackCNN
    config: TrainConfig
    history: pd.DataFrame
    batches: pd.DataFrame
    best_epoch: int | None
    selected_epoch: int
    seconds: float


def train_cnn(
    patches: torch.Tensor,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    validation: tuple[torch.Tensor, np.ndarray] | None = None,
    verbose: bool = True,
) -> TrainResult:
    """SGD sem momentum; a perda segue 1/(2N) * soma dos erros quadráticos.

    Registra a perda por minibatch, a média móvel rL do MATLAB e métricas ao
    fim de cada época. No último lote, permite menos de batch_size exemplos.
    """
    if config.epochs < 1 or config.batch_size < 1 or config.learning_rate <= 0:
        raise ValueError("Épocas, batch_size e learning_rate devem ser positivos.")
    if config.weight_decay < 0 or config.min_delta < 0:
        raise ValueError("weight_decay e min_delta não podem ser negativos.")
    if config.patience is not None and config.patience < 1:
        raise ValueError("patience deve ser um inteiro positivo ou None.")
    if (config.patience is not None or config.restore_best) and validation is None:
        raise ValueError("Early stopping e restore_best exigem dados de validação.")
    labels = np.array(labels, copy=True)
    if labels.ndim != 1 or len(patches) != len(labels) or len(labels) == 0:
        raise ValueError("Quantidade de patches e rótulos incompatível.")
    if not np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))):
        raise ValueError("Os rótulos precisam ser inteiros consecutivos começando em zero.")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = CrackCNN(len(np.unique(labels)), config.channels, config.kernel_size).to(device)
    x = patches.to(device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    if validation is not None:
        vx = validation[0].to(device)
        vy = torch.tensor(np.array(validation[1], copy=True), dtype=torch.long, device=device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(config.seed)
    rows, batch_rows = [], []
    smoothed = None
    best_loss, best_epoch, best_weights, waiting = float("inf"), None, None, 0
    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for batch_number, begin in enumerate(range(0, len(x), config.batch_size), 1):
            index = order[begin:begin + config.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = matlab_loss(model(x[index]), y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Perda não finita durante o treinamento.")
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            smoothed = value if smoothed is None else 0.99 * smoothed + 0.01 * value
            batch_rows.append({
                "epoch": epoch, "batch": batch_number, "loss": value, "rL": smoothed,
            })
        model.eval()
        with torch.inference_mode():
            scores = model(x)
            row = {
                "epoch": epoch,
                "train_loss": float(matlab_loss(scores, y).cpu()),
                "train_accuracy": float((scores.argmax(1) == y).float().mean().cpu()),
            }
            if validation is not None:
                scores = model(vx)
                row.update({
                    "validation_loss": float(matlab_loss(scores, vy).cpu()),
                    "validation_accuracy": float((scores.argmax(1) == vy).float().mean().cpu()),
                })
                if row["validation_loss"] < best_loss - config.min_delta:
                    best_loss, best_epoch, waiting = row["validation_loss"], epoch, 0
                    if config.restore_best:
                        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    waiting += 1
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError("Métrica não finita ao avaliar o treinamento.")
        rows.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            suffix = f" | val_loss={row['validation_loss']:.6f}" if validation is not None else ""
            print(f"Época {epoch:3d}/{config.epochs} | loss={row['train_loss']:.6f}"
                  f" | acurácia={row['train_accuracy']:.3f}{suffix}")
        if config.patience is not None and waiting >= config.patience:
            if verbose:
                print(f"Early stopping na época {epoch}; melhor época: {best_epoch}.")
            break
    selected_epoch = rows[-1]["epoch"]
    if config.restore_best:
        model.load_state_dict(best_weights)
        selected_epoch = best_epoch
    if device.type == "cuda":
        torch.cuda.synchronize()
    return TrainResult(
        model=model, config=config, history=pd.DataFrame(rows), batches=pd.DataFrame(batch_rows),
        best_epoch=best_epoch, selected_epoch=selected_epoch, seconds=perf_counter() - started,
    )


def classification_metrics(
    model: CrackCNN, patches: torch.Tensor, labels: np.ndarray, classes: dict[int, str],
    subset: str, batch_size: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Métricas macro incluem todas as classes; precisão/F1 indefinidas valem 0."""
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        scores = torch.cat([
            model(patches[i:i + batch_size].to(device)).cpu()
            for i in range(0, len(patches), batch_size)
        ])
    prediction = scores.argmax(1).numpy()
    labels = np.asarray(labels, dtype=int)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (labels, prediction), 1)
    per_class = []
    for label, name in classes.items():
        tp = int(matrix[label, label])
        support, predicted = int(matrix[label].sum()), int(matrix[:, label].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "subset": subset, "label": label, "class": name, "support": support,
            "precision": precision, "recall": recall, "f1": f1,
        })
    per_class = pd.DataFrame(per_class)
    summary = pd.DataFrame([{
        "subset": subset, "samples": len(labels),
        "loss": float(matlab_loss(scores, torch.tensor(labels)).item()),
        "accuracy": float(np.mean(prediction == labels)),
        "precision_macro": per_class.precision.mean(),
        "recall_macro": per_class.recall.mean(), "f1_macro": per_class.f1.mean(),
    }])
    confusion = pd.DataFrame(matrix, index=list(classes.values()), columns=list(classes.values()))
    confusion.index.name, confusion.columns.name = "true_class", "predicted_class"
    return summary, per_class, confusion


def classify_image(
    model: CrackCNN, extractor: PatchExtractor, batch_size: int = 512,
    stride: int = 1, verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Classifica apenas a grade solicitada; não interpola rótulos não calculados.

    stride=1 percorre todos os pixels, em ordem de linhas NumPy. stride>1 é
    uma prévia em grade reduzida, com os eixos x,z exportados explicitamente.
    """
    if batch_size < 1 or stride < 1:
        raise ValueError("batch_size e stride devem ser positivos.")
    device = next(model.parameters()).device
    x_axis = np.arange(0, extractor.shape[1], stride)
    z_axis = np.arange(0, extractor.shape[0], stride)
    total = len(x_axis) * len(z_axis)
    score_map = np.empty((total, model.classes), dtype=np.float32)
    model.eval()
    started = perf_counter()
    report_every = max(1, (total + batch_size - 1) // batch_size // 5)
    with torch.inference_mode():
        for batch, begin in enumerate(range(0, total, batch_size)):
            flat = np.arange(begin, min(begin + batch_size, total))
            z = z_axis[flat // len(x_axis)]
            x = x_axis[flat % len(x_axis)]
            patches = extractor.at(x, z).to(device)
            score_map[flat] = model(patches).cpu().numpy()
            if verbose and (batch % report_every == 0 or flat[-1] == total - 1):
                print(f"Classificação: {flat[-1] + 1:,}/{total:,} centros")
    score_map = score_map.reshape(len(z_axis), len(x_axis), model.classes)
    if not np.isfinite(score_map).all():
        raise FloatingPointError("Escores não finitos na classificação da imagem.")
    label_map = score_map.argmax(axis=2).astype(np.uint8)
    return label_map, score_map, x_axis, z_axis, perf_counter() - started


def export_run(
    directory: Path, result: TrainResult, annotations: pd.DataFrame,
    training_indices: np.ndarray, validation_indices: np.ndarray,
    classes: dict[int, str], metrics: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(directory / "history.csv", index=False)
    result.batches.to_csv(directory / "batches.csv", index=False)
    table = annotations.copy()
    table["subset"] = "unused"
    table.loc[training_indices, "subset"] = "training"
    table.loc[validation_indices, "subset"] = "validation"
    table.to_csv(directory / "annotations.csv", index=False)
    pd.concat([m[0] for m in metrics], ignore_index=True).to_csv(directory / "metrics.csv", index=False)
    pd.concat([m[1] for m in metrics], ignore_index=True).to_csv(directory / "class_metrics.csv", index=False)
    for summary, _, matrix in metrics:
        matrix.to_csv(directory / f"confusion_{summary.iloc[0]['subset']}.csv")
    metadata = {
        "config": asdict(result.config), "classes": classes,
        "parameters": sum(p.numel() for p in result.model.parameters()),
        "epochs_run": len(result.history), "selected_epoch": result.selected_epoch,
        "best_validation_epoch": result.best_epoch, "training_seconds": result.seconds,
        "source_url": LAB_URL, "zip_sha256": ZIP_SHA256,
        "device": str(next(result.model.parameters()).device),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "cuda": torch.version.cuda, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    (directory / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
        "config": asdict(result.config), "classes": classes,
        "selected_epoch": result.selected_epoch,
    }, directory / "model.pt")
