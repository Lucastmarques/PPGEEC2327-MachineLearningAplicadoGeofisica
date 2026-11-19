"""Experimentos compartilhados pelo notebook e pela execução de terminal."""

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy

from .csc import learn_dictionary, preprocess, snr_db, sparse_reconstruct
from .data import ARCHIVE_SHA256, ARCHIVE_URL, load_original_data

SIZES = ((11, 7), (21, 11), (31, 15))
COUNTS = (1, 3, 6, 9, 12)
BETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


@dataclass(frozen=True)
class ExperimentConfig:
    training_seed: int = 2026
    noise_seed: int = 2027
    noise_sigma: float = 0.3
    training_beta: float = 1.0
    denoising_beta: float = 0.5
    lambda_residual: float = 1.0
    train_max_iter: int = 80
    denoise_max_iter: int = 80
    inner_iter: int = 10
    tol: float = 1e-3
    activity_relative_threshold: float = 1e-3

    def __post_init__(self):
        if not np.isfinite([self.noise_sigma, self.activity_relative_threshold]).all() or self.noise_sigma <= 0 or not 0 < self.activity_relative_threshold < 1:
            raise ValueError("Escala do ruído e limiar de atividade inválidos.")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ExperimentSuite:
    """Estado de uma execução; nunca reutiliza modelos de execuções anteriores.

    As chamadas dentro da mesma execução reutilizam o dicionário base em
    memória. Uma nova instância sempre calcula novos resultados com sua
    configuração e atualiza o diretório de saída selecionado.
    """

    def __init__(self, root, config=None, output_dir=None, *, download=True):
        self.root = Path(root).resolve()
        self.config = config or ExperimentConfig()
        self.output = Path(output_dir).resolve() if output_dir else self.root / "original" / "results" / "main"
        self.gathers, self.matlab_reference = load_original_data(self.root, download=download)
        self.train = preprocess(self.gathers[:100, :, 0])
        self.clean = preprocess(self.gathers[:100, :, 1])
        self.noisy = self.clean + self.config.noise_sigma * np.random.default_rng(self.config.noise_seed).standard_normal(self.clean.shape)
        self.trainings, self.denoisings, self.metrics = {}, {}, {}
        self.training_shapes = {}
        self.beta_experiments = {}
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "data").mkdir(exist_ok=True)
        np.savez_compressed(self.output / "data/preprocessed.npz", train=self.train, clean=self.clean, noisy=self.noisy)
        self.metadata = {
            "started_at": datetime.now(ZoneInfo("America/Fortaleza")).isoformat(),
            "configuration": asdict(self.config),
            "archive_url": ARCHIVE_URL, "archive_sha256": ARCHIVE_SHA256,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
            "data_sha256": {name: sha256(array.tobytes()).hexdigest() for name, array in [("train", self.train), ("clean", self.clean), ("noisy", self.noisy)]},
            "training_csg_matlab_index": 1, "evaluation_csg_matlab_index": 2,
        }
        save_json(self.output / "settings.json", self.metadata)

    def train_dictionary(self, name, shape=(21, 11), count=3):
        if name in self.trainings:
            if self.training_shapes[name] != (*shape, count):
                raise ValueError("O identificador já pertence a outra configuração.")
            return self.trainings[name]
        c = self.config
        result = learn_dictionary(self.train, shape, count, beta=c.training_beta, lambda_residual=c.lambda_residual, max_iter=c.train_max_iter, inner_iter=c.inner_iter, tol=c.tol, seed=c.training_seed)
        self.trainings[name] = result
        self.training_shapes[name] = (*shape, count)
        directory = self.output / "training" / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "state.npz", filters=result.filters, full_filters=result.full_filters, coefficients=result.coefficients, reconstruction=result.reconstruction, reconstruction_extended=result.reconstruction_extended, filter_history=result.filter_history)
        pd.DataFrame(result.history).to_csv(directory / "history.csv", index=False)
        save_json(directory / "config.json", {"filter_shape": list(shape), "n_filters": count, **asdict(c), "objective": result.objective, "iterations": result.iterations, "attempted_iterations": result.attempted_iterations, "stop_reason": result.stop_reason, "elapsed_seconds": result.elapsed_seconds, "filter_norms": np.linalg.norm(result.filters, axis=(0, 1)).tolist()})
        print(f"Treino {name}: {shape}, K={count}, {result.iterations} iterações aceitas, {result.stop_reason}, {result.elapsed_seconds:.2f} s", flush=True)
        return result

    def denoise(self, name, dictionary="reference", beta=None):
        if name in self.denoisings:
            raise ValueError(f"Resultado {name} já calculado nesta execução; use um novo identificador.")
        c = self.config
        beta = c.denoising_beta if beta is None else beta
        filters = self.matlab_reference["d"] if dictionary == "matlab_supplied" else self.trainings[dictionary].filters
        result = sparse_reconstruct(self.noisy, filters, beta=beta, lambda_residual=c.lambda_residual, max_iter=c.denoise_max_iter, tol=c.tol)
        self.denoisings[name] = result
        error = result.reconstruction - self.clean
        threshold = c.activity_relative_threshold * float(np.max(np.abs(result.coefficients)))
        metrics = {
            "experiment": name, "dictionary": dictionary,
            "filter_height": filters.shape[0], "filter_width": filters.shape[1], "n_filters": filters.shape[2],
            "beta": float(beta), "snr_before_db": snr_db(self.clean, self.noisy),
            "snr_after_db": snr_db(self.clean, result.reconstruction),
            "mse": float(np.mean(error**2)),
            "relative_l2_error": float(np.sqrt(np.sum(error**2) / np.sum(self.clean**2))),
            "active_fraction": float(np.mean(np.abs(result.coefficients) > threshold)),
            "activity_threshold": threshold,
            "removed_energy_fraction": float(np.sum((self.noisy - result.reconstruction)**2) / np.sum(self.noisy**2)),
            "objective": result.objective, "iterations": result.iterations,
            "stop_reason": result.stop_reason, "denoise_seconds": result.elapsed_seconds,
            "training_seconds": 0.0 if dictionary == "matlab_supplied" else self.trainings[dictionary].elapsed_seconds,
            "max_filter_norm": float(np.linalg.norm(filters, axis=(0, 1)).max()),
        }
        metrics["snr_gain_db"] = metrics["snr_after_db"] - metrics["snr_before_db"]
        self.metrics[name] = metrics
        directory = self.output / "denoising" / name
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "state.npz", filters=filters, coefficients=result.coefficients, reconstruction=result.reconstruction, residual=self.noisy - result.reconstruction)
        pd.DataFrame(result.history).to_csv(directory / "history.csv", index=False)
        save_json(directory / "metrics.json", metrics)
        self.export_summary()
        print(f"Denoising {name}: beta={beta:g}, SNR={metrics['snr_after_db']:.3f} dB, {result.elapsed_seconds:.2f} s", flush=True)
        return result

    def export_summary(self):
        self.summary.to_csv(self.output / "summary.csv", index=False)
        rows = [{"dictionary": name, "filter_height": self.training_shapes[name][0], "filter_width": self.training_shapes[name][1], "n_filters": self.training_shapes[name][2], "objective": result.objective, "iterations": result.iterations, "attempted_iterations": result.attempted_iterations, "stop_reason": result.stop_reason, "elapsed_seconds": result.elapsed_seconds, "max_filter_norm": float(np.linalg.norm(result.filters, axis=(0, 1)).max())} for name, result in self.trainings.items()]
        pd.DataFrame(rows).to_csv(self.output / "training_summary.csv", index=False)

    @property
    def summary(self):
        return pd.DataFrame(self.metrics.values())


def size_name(shape):
    return "reference" if tuple(shape) == (21, 11) else f"size_{shape[0]}x{shape[1]}"


def count_name(count):
    return "reference" if count == 3 else f"count_{count}"


def beta_name(beta):
    return "beta_" + f"{beta:g}".replace(".", "p")


def run_count_sweep(suite):
    """Varia K no suporte do modelo de referência e reutiliza seu caso K=3."""
    shape = suite.training_shapes["reference"][:2]
    names = []
    for count in COUNTS:
        name = count_name(count)
        suite.train_dictionary(name, shape, count)
        if name not in suite.denoisings:
            suite.denoise(name, name)
        names.append(name)
    suite.metadata["count_sweep"] = {
        "filter_shape": list(shape),
        "n_filters": list(COUNTS),
        "experiments": names,
    }
    save_json(suite.output / "settings.json", suite.metadata)
    return names


def run_beta_sweep(suite):
    """Usa a maior SNR da série de K; em empate, mantém a ordem de COUNTS.

    A reconstrução já calculada no beta de referência é reutilizada junto
    com seu identificador. Todos os demais betas usam o mesmo dicionário.
    """
    candidates = [count_name(count) for count in COUNTS]
    if any(name not in suite.metrics for name in candidates):
        raise ValueError("Execute toda a série de quantidade de filtros antes da penalização.")
    baseline = max(candidates, key=lambda name: suite.metrics[name]["snr_after_db"])
    dictionary = suite.metrics[baseline]["dictionary"]
    for beta in BETAS:
        name = baseline if beta == suite.config.denoising_beta else beta_name(beta)
        if beta != suite.config.denoising_beta:
            suite.denoise(name, dictionary=dictionary, beta=beta)
        suite.beta_experiments[beta] = name
    suite.metadata["beta_sweep"] = {
        "selection_metric": "snr_after_db",
        "candidate_experiments": candidates,
        "dictionary": dictionary,
        "experiments": [{"beta": beta, "experiment": name} for beta, name in suite.beta_experiments.items()],
    }
    save_json(suite.output / "settings.json", suite.metadata)
    return list(suite.beta_experiments.values())


def run_full_experiment(suite):
    """Mesma grade utilizada nas seções individuais do notebook."""
    suite.train_dictionary("reference")
    suite.denoise("reference")
    suite.denoise("matlab_supplied", "matlab_supplied")
    for shape in SIZES:
        name = size_name(shape)
        if name != "reference":
            suite.train_dictionary(name, shape, 3)
            suite.denoise(name, name)
    run_count_sweep(suite)
    run_beta_sweep(suite)
    suite.export_summary()
    return suite
