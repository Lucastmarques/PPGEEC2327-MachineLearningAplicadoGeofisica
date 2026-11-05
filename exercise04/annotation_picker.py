"""Seleção manual não bloqueante para JupyterLab e VS Code, como no exercise03."""
from pathlib import Path

import numpy as np
import pandas as pd

from crack_cnn import validate_annotations


class AnnotationPicker:
    """Seleciona uma classe por vez e salva x,z,label com índices começando em 0."""

    def __init__(
        self, image: np.ndarray, path: Path, classes: dict[int, str], counts: dict[int, int],
        order: tuple[int, ...] | None = None,
    ):
        try:
            import ipywidgets as widgets
            import plotly.graph_objects as go
            from IPython.display import display
        except ImportError as error:
            raise RuntimeError(
                "Instale plotly, anywidget e ipywidgets pelo conda.yaml e reinicie o kernel."
            ) from error
        self.path, self.classes, self.counts = Path(path), classes, counts
        self.order = tuple(classes) if order is None else order
        if set(self.order) != set(classes) or set(counts) != set(classes) or min(counts.values()) < 1:
            raise ValueError("As classes, as contagens e a ordem de seleção devem corresponder.")
        self.image_shape = image.shape
        self.points: list[tuple[int, int, int]] = []
        self.saved = False
        self._display = display
        colors = ["#00bfff", "#ffcc00", "#ff4b82", "#66ff66"]
        self.figure = go.FigureWidget(data=[go.Heatmap(
            z=image, x=np.arange(image.shape[1]), y=np.arange(image.shape[0]),
            colorscale="gray", showscale=False,
            hovertemplate="x=%{x}<br>z=%{y}<extra></extra>",
        )])
        for i, (label, name) in enumerate(classes.items()):
            self.figure.add_trace(go.Scatter(
                x=[], y=[], mode="markers", name=f"{label}: {name}",
                marker={"color": colors[i % len(colors)], "size": 7,
                        "symbol": "circle-open", "line": {"width": 1.5}},
                hoverinfo="skip",
            ))
        self.figure.update_layout(
            width=1100, height=600, title=f"Anotações — {self.path.name}",
            margin={"l": 50, "r": 20, "t": 80, "b": 45},
            xaxis={"title": "x (coluna/pixel)", "constrain": "domain"},
            yaxis={"title": "z (linha/pixel)", "autorange": "reversed", "scaleanchor": "x"},
            legend={"orientation": "h", "y": 1.12}, clickmode="event",
        )
        self.figure.data[0].on_click(self._on_click)
        self.status = widgets.HTML()
        undo = widgets.Button(description="Desfazer", icon="undo")
        clear = widgets.Button(description="Limpar tudo", icon="trash")
        self.save_button = widgets.Button(description="Salvar CSV", icon="save", button_style="success")
        undo.on_click(self._undo)
        clear.on_click(self._clear)
        self.save_button.on_click(self._save)
        self.container = widgets.VBox([
            self.status, widgets.HBox([undo, clear, self.save_button]), self.figure,
        ])
        self._refresh()

    def _active_class(self):
        for label in self.order:
            if sum(p[2] == label for p in self.points) < self.counts[label]:
                return label
        return None

    def _refresh(self, message: str = ""):
        counts = {label: sum(p[2] == label for p in self.points) for label in self.classes}
        active = self._active_class()
        prompt = (f"Selecione: <b>{self.classes[active]}</b>. " if active is not None
                  else "Seleção completa. Revise e clique em <b>Salvar CSV</b>. ")
        progress = " | ".join(
            f"{self.classes[k]}: {v}/{self.counts[k]}" for k, v in counts.items()
        )
        self.status.value = f"{prompt}{progress}<br>{message}"
        self.save_button.disabled = active is not None
        with self.figure.batch_update():
            for i, label in enumerate(self.classes, 1):
                chosen = [(x, z) for x, z, y in self.points if y == label]
                self.figure.data[i].x = [p[0] for p in chosen]
                self.figure.data[i].y = [p[1] for p in chosen]

    def _on_click(self, trace, points, state):
        active = self._active_class()
        if active is None or not points.xs or not points.ys:
            return
        x, z = int(np.floor(points.xs[0] + 0.5)), int(np.floor(points.ys[0] + 0.5))
        if not (0 <= x < self.image_shape[1] and 0 <= z < self.image_shape[0]):
            return
        if any(px == x and pz == z for px, pz, _ in self.points):
            self._refresh("Este pixel já foi selecionado. Escolha outro.")
            return
        self.points.append((x, z, active))
        self.saved = False
        self._refresh()

    def _undo(self, _):
        if self.points:
            self.points.pop()
        self.saved = False
        self._refresh()

    def _clear(self, _):
        self.points.clear()
        self.saved = False
        self._refresh()

    def _save(self, _):
        try:
            table = validate_annotations(
                pd.DataFrame(self.points, columns=["x", "z", "label"]),
                self.image_shape, self.classes, self.counts, allow_duplicates=False,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Criação exclusiva: inclusive um widget antigo não pode substituir
            # um CSV criado depois que ele foi aberto.
            with self.path.open("x", encoding="utf-8", newline="") as stream:
                table.to_csv(stream, index=False)
            self.saved = True
            self._refresh("<b>CSV salvo. Execute a próxima célula.</b>")
        except FileExistsError:
            self._refresh("<b>O CSV já existe e foi preservado. Reexecute a seção para carregá-lo.</b>")
        except (ValueError, OSError) as error:
            self._refresh(f"Não foi possível salvar: {error}")

    def show(self):
        self._display(self.container)
