"""Figuras estáticas compartilhadas pelo notebook e pelo relatório exportado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .csc import crop_observed, fft2, ifft2
from .experiments import BETAS, COUNTS, SIZES, count_name, size_name


def _image(ax, image, title, limit, *, cmap="gray", equal=False):
    artist = ax.imshow(image, cmap=cmap, vmin=-limit, vmax=limit, aspect="equal" if equal else "auto", interpolation="nearest")
    ax.set(title=title, xlabel="Traço", ylabel="Amostra")
    return artist


def plot_gathers(gathers):
    fig, axes = plt.subplots(1, 5, figsize=(15, 5), layout="constrained")
    limit = float(np.abs(gathers).max())
    for i, ax in enumerate(axes):
        artist = _image(ax, gathers[:, :, i], f"CSG {i + 1}", limit)
        ax.axhline(99.5, color="tab:red", linewidth=1, label="Recorte de 100 amostras")
    fig.colorbar(artist, ax=axes, label="Amplitude bruta", shrink=.75)
    fig.suptitle("Cinco CSGs fornecidos — escala comum")
    return fig


def plot_preprocessing(suite):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for row, index in enumerate((0, 1)):
        raw = suite.gathers[:100, :, index]
        normalized = suite.train if row == 0 else suite.clean
        raw_artist = _image(axes[row, 0], raw, f"CSG {index + 1}: bruto", float(np.abs(suite.gathers[:100, :, :2]).max()))
        norm_artist = _image(axes[row, 1], normalized, f"CSG {index + 1}: normalizado e centralizado", float(max(np.abs(suite.train).max(), np.abs(suite.clean).max())))
    fig.colorbar(raw_artist, ax=axes[:, 0], label="Amplitude bruta", shrink=.8)
    fig.colorbar(norm_artist, ax=axes[:, 1], label="Amplitude normalizada", shrink=.8)
    return fig


def plot_filters(filters, title="Filtros aprendidos"):
    k = filters.shape[2]
    columns = min(k, 6)
    rows = (k + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(max(5, 1.9 * columns), 3.4 * rows), layout="constrained", squeeze=False)
    limit = max(float(np.abs(filters).max()), np.finfo(float).eps)
    for i, ax in enumerate(axes.ravel()):
        if i >= k:
            ax.axis("off")
            continue
        image = ax.imshow(filters[:, :, i], cmap="gray", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=f"d{i + 1} · ‖d‖={np.linalg.norm(filters[:, :, i]):.3f}", xlabel="Coluna", ylabel="Linha")
    fig.colorbar(image, ax=axes.ravel(), label="Peso", shrink=.7)
    fig.suptitle(f"{title} — {filters.shape[0]} × {filters.shape[1]}, K={k}")
    return fig


def plot_training(result, tol=1e-3):
    history = result.history
    x = [row["iteration"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].semilogy(x, [r["objective_d"] for r in history], label="Após atualizar D")
    axes[0].semilogy(x, [r["objective_z"] for r in history], label="Após atualizar Z")
    axes[0].set(xlabel="Iteração externa aceita", ylabel="Objetivo", title="Treinamento CSC")
    if len(history) > 1:
        axes[1].semilogy(x[1:], [max(r["relative_d"], 1e-16) for r in history[1:]], label="Mudança de D")
        axes[1].semilogy(x[1:], [max(r["relative_z"], 1e-16) for r in history[1:]], label="Mudança de Z")
    axes[1].axhline(tol, color="black", linestyle="--", label="Tolerância")
    axes[1].set(xlabel="Iteração externa aceita", ylabel="Mudança relativa", title=f"Parada: {result.stop_reason}")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    return fig


def plot_decomposition(result, image_shape):
    filters, z = result.filters, result.coefficients[..., 0]
    parts = crop_observed(ifft2(fft2(result.full_filters) * fft2(z)), image_shape, filters.shape[:2])
    maps = crop_observed(z, image_shape, filters.shape[:2])
    fig, axes = plt.subplots(filters.shape[2], 3, figsize=(10, 2.6 * filters.shape[2]), layout="constrained", squeeze=False)
    d_limit = max(float(np.abs(filters).max()), 1e-15)
    z_limit, part_limit = max(float(np.abs(maps).max()), 1e-15), max(float(np.abs(parts).max()), 1e-15)
    for i in range(filters.shape[2]):
        axes[i, 0].imshow(filters[:, :, i], cmap="gray", vmin=-d_limit, vmax=d_limit)
        axes[i, 0].set(title=f"Filtro d{i + 1}", xlabel="Coluna", ylabel="Linha")
        axes[i, 1].imshow(np.abs(maps[:, :, i]), cmap="magma", vmin=0, vmax=z_limit, aspect="auto")
        axes[i, 1].set(title=f"|z{i + 1}| (região observada)", xlabel="Traço", ylabel="Amostra")
        _image(axes[i, 2], parts[:, :, i], f"Contribuição d{i + 1} * z{i + 1}", part_limit)
    fig.suptitle("Decomposição do treinamento (convolução com o estado estendido do ADMM)")
    return fig


def plot_denoising(suite, name="reference"):
    result, metrics = suite.denoisings[name], suite.metrics[name]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    limit = float(np.abs(suite.noisy).max())
    images = [suite.clean, suite.noisy, result.reconstruction, suite.noisy - result.reconstruction]
    titles = ["Referência normalizada (CSG 2)", f"Com ruído · {metrics['snr_before_db']:.2f} dB", f"Reconstrução · {metrics['snr_after_db']:.2f} dB", "Residual: ruidoso − reconstrução"]
    for ax, array, title in zip(axes.ravel(), images, titles):
        artist = _image(ax, array, title, limit)
    fig.colorbar(artist, ax=axes.ravel(), label="Amplitude normalizada", shrink=.8)
    fig.suptitle(f"{name} · beta={metrics['beta']:g} · escala comum")
    return fig


def plot_comparison(suite, names, title):
    fig, axes = plt.subplots(len(names), 2, figsize=(9, 2.8 * len(names)), layout="constrained", squeeze=False)
    limit = float(np.abs(suite.noisy).max())
    for row, name in enumerate(names):
        result, m = suite.denoisings[name], suite.metrics[name]
        label = f"{m['filter_height']}×{m['filter_width']}, K={m['n_filters']}, beta={m['beta']:g}"
        artist = _image(axes[row, 0], result.reconstruction, f"{label} · SNR={m['snr_after_db']:.2f} dB", limit)
        _image(axes[row, 1], suite.noisy - result.reconstruction, "Residual", limit)
    fig.colorbar(artist, ax=axes.ravel(), label="Amplitude normalizada", shrink=.8)
    fig.suptitle(title + " — mesmo dado ruidoso e mesma escala")
    return fig


def plot_sweeps(suite):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    size_rows = [suite.metrics[size_name(s)] for s in SIZES]
    count_rows = [suite.metrics[count_name(k)] for k in COUNTS]
    beta_rows = [suite.metrics[suite.beta_experiments[b]] for b in BETAS]
    series = [(axes[0], [f"{s[0]}×{s[1]}" for s in SIZES], size_rows, "Tamanho (K=3)"), (axes[1], list(COUNTS), count_rows, "Quantidade (21×11)"), (axes[2], list(BETAS), beta_rows, "Penalização L1 (dicionário fixo)")]
    for ax, x, rows, title in series:
        ax.plot(x, [r["snr_after_db"] for r in rows], "o-", label="Após CSC")
        ax.axhline(suite.metrics["reference"]["snr_before_db"], color="gray", linestyle="--", label="Entrada")
        ax.set(title=title, ylabel="SNR (dB)")
        ax.grid(alpha=.25)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("beta")
    axes[0].legend()
    return fig


def plot_cnn_comparison(csc_filters, cnn_path):
    with np.load(cnn_path, allow_pickle=False) as data:
        cnn_filters = data["filters"]
    ncols = max(csc_filters.shape[2], cnn_filters.shape[2])
    fig, axes = plt.subplots(2, ncols, figsize=(2 * ncols, 5.5), layout="constrained", squeeze=False)
    for row, (bank, label) in enumerate(((csc_filters, "CSC sísmico"), (cnn_filters, "CNN de fraturas"))):
        limit = max(float(np.abs(bank).max()), 1e-15)
        for i, ax in enumerate(axes[row]):
            if i >= bank.shape[2]:
                ax.axis("off")
                continue
            ax.imshow(bank[:, :, i], cmap="gray", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set(title=f"{label} · {i + 1}\n{bank.shape[0]}×{bank.shape[1]}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Filtros treinados em domínios distintos — escala de pesos própria de cada banco")
    return fig


def save_report_figures(suite):
    """Exporta um conjunto compacto de figuras sem manter canvases abertos."""
    directory = suite.output / "figures"
    directory.mkdir(exist_ok=True)
    factories = {
        "gathers": lambda: plot_gathers(suite.gathers),
        "preprocessing": lambda: plot_preprocessing(suite),
        "reference_filters": lambda: plot_filters(suite.trainings["reference"].filters),
        "training_convergence": lambda: plot_training(suite.trainings["reference"], suite.config.tol),
        "training_decomposition": lambda: plot_decomposition(suite.trainings["reference"], suite.train.shape),
        "reference_denoising": lambda: plot_denoising(suite),
        "size_comparison": lambda: plot_comparison(suite, [size_name(s) for s in SIZES], "Variação do tamanho"),
        "count_comparison": lambda: plot_comparison(suite, [count_name(k) for k in COUNTS], "Variação de K"),
        "beta_comparison": lambda: plot_comparison(suite, [suite.beta_experiments[b] for b in (.1, .5, 5)], "Variação de beta"),
        "snr_sweeps": lambda: plot_sweeps(suite),
    }
    cnn_path = suite.root / "cnn_reference.npz"
    if cnn_path.exists():
        factories["csc_cnn_filters"] = lambda: plot_cnn_comparison(suite.trainings["reference"].filters, cnn_path)
    for name, factory in factories.items():
        fig = factory()
        fig.savefig(directory / f"{name}.png", dpi=150)
        plt.close(fig)
    return directory
