"""CSC com ADMM, em precisão dupla e eixos espaciais (0, 1).

Adaptado dos solvers MATLAB de Fast and Flexible Convolutional Sparse Coding,
F. Heide, W. Heidrich e G. Wetzstein (CVPR 2015), modificados por Zhaolun Liu
em 28/10/2018 para o laboratório da University of Utah/KAUST.

Filtros compactos: (h, w, K); coeficientes de treinamento: (Hext, Wext, K, N).
Os sistemas por frequência são escritos com eixos explícitos, eliminando a
necessidade de reproduzir os reshape/permute por colunas do MATLAB.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.signal import convolve2d


def _real_array(value, ndim=None):
    if np.iscomplexobj(value):
        raise ValueError("Os dados espaciais devem ser reais.")
    array = np.asarray(value, dtype=np.float64)
    if (ndim is not None and array.ndim != ndim) or not array.size or not np.isfinite(array).all():
        raise ValueError(f"Array deve ser finito, não vazio e ter {ndim or 'dimensões válidas'} dimensões.")
    return array


def _odd_shape(shape):
    if len(shape) != 2 or any(int(s) != s or s < 1 or s % 2 != 1 for s in shape):
        raise ValueError("As duas dimensões espaciais do filtro devem ser inteiras, positivas e ímpares.")
    return tuple(map(int, shape))


def _parameters(data, beta, lambda_residual, max_iter, tol):
    if not np.isfinite([beta, lambda_residual, tol]).all() or beta <= 0 or lambda_residual <= 0 or tol < 0:
        raise ValueError("beta e lambda_residual devem ser positivos; tol deve ser não negativa.")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter deve ser inteiro positivo.")
    if data.max() <= 0:
        raise ValueError("A heurística ADMM do MATLAB exige max(dados) > 0.")


def fft2(array):
    """Transforma apenas os eixos espaciais, incluindo arrays com K e N."""
    return np.fft.fft2(array, axes=(0, 1))


def ifft2(array):
    return np.fft.ifft2(array, axes=(0, 1)).real


def soft_threshold(array, threshold):
    """Operador proximal L1, definido também quando a entrada é zero."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("O limiar deve ser finito e não negativo.")
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)


def local_contrast_normalization(data, window=21, sigma=4.773):
    """Reproduz fspecial + rconv2 e a estatística de ordem do MATLAB.

    A reflexão não repete o pixel da borda. Para N par, usa o elemento
    central inferior, em vez da média dos dois elementos centrais.
    """
    data = _real_array(data)
    if data.ndim not in (2, 3):
        raise ValueError("Use uma imagem 2D ou um conjunto (H, W, N).")
    _odd_shape((window, window))
    if not np.isfinite(sigma) or sigma <= 0 or window // 2 >= min(data.shape[:2]):
        raise ValueError("Janela incompatível com os dados ou sigma inválido.")
    radius = window // 2
    coords = np.arange(-radius, radius + 1)
    kernel = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * sigma**2))
    kernel[kernel < np.finfo(float).eps * kernel.max()] = 0
    kernel /= kernel.sum()
    images = data[..., None] if data.ndim == 2 else data
    output = np.empty_like(images)
    for i in range(images.shape[2]):
        image = images[:, :, i]
        mean = convolve2d(np.pad(image, radius, mode="reflect"), kernel, mode="valid")
        mean_square = convolve2d(np.pad(image**2, radius, mode="reflect"), kernel, mode="valid")
        std = np.sqrt(np.maximum(mean_square - mean**2, 0.0))
        flat = std.ravel()
        middle = (flat.size - 1) // 2
        floor = np.partition(flat, middle)[middle]
        if floor == 0:
            nonzero = flat[flat != 0]
            if nonzero.size:
                middle = (nonzero.size - 1) // 2
                floor = np.partition(nonzero, middle)[middle]
        denominator = np.maximum(std, floor)
        denominator[denominator == 0] = np.finfo(float).eps
        output[:, :, i] = (image - mean) / denominator
    return output[:, :, 0] if data.ndim == 2 else output


def preprocess(data, window=21):
    normalized = local_contrast_normalization(data, window)
    return normalized - normalized.mean(axis=(0, 1), keepdims=True)


def embed_filters(filters, spatial_shape):
    """Preenchimento à direita e deslocamento da origem, como psf2otf."""
    filters = _real_array(filters, 3)
    h, w = _odd_shape(filters.shape[:2])
    if len(spatial_shape) != 2 or any(int(s) != s for s in spatial_shape) or h > spatial_shape[0] or w > spatial_shape[1]:
        raise ValueError("O domínio estendido deve comportar os filtros.")
    padded = np.zeros((*map(int, spatial_shape), filters.shape[2]))
    padded[:h, :w] = filters
    return np.roll(padded, (-(h // 2), -(w // 2)), axis=(0, 1))


def compact_filters(full_filters, filter_shape):
    h, w = _odd_shape(filter_shape)
    return np.roll(full_filters, (h // 2, w // 2), axis=(0, 1))[:h, :w].copy()


def project_filters(full_filters, filter_shape):
    """Projeção do suporte e da norma L2 de cada filtro na bola unitária."""
    compact = compact_filters(full_filters, filter_shape)
    norms = np.sqrt(np.sum(compact**2, axis=(0, 1), keepdims=True))
    compact /= np.maximum(norms, 1.0)
    return embed_filters(compact, full_filters.shape[:2])


def crop_observed(array, image_shape, filter_shape):
    h, w = _odd_shape(filter_shape)
    return array[h // 2:h // 2 + image_shape[0], w // 2:w // 2 + image_shape[1], ...]


def convolution_components(filters, coefficients, image_shape):
    """Uma imagem por filtro; a soma é a reconstrução com filtros compactos."""
    filters = _real_array(filters, 3)
    coefficients = _real_array(coefficients, 3)
    expected = (image_shape[0] + filters.shape[0] - 1, image_shape[1] + filters.shape[1] - 1, filters.shape[2])
    if coefficients.shape != expected:
        raise ValueError(f"Coeficientes devem ter forma {expected}.")
    extended = ifft2(fft2(embed_filters(filters, coefficients.shape[:2])) * fft2(coefficients))
    return crop_observed(extended, image_shape, filters.shape[:2])


def relative_change(current, previous):
    numerator = np.sqrt(np.sum(np.abs(current - previous)**2))
    denominator = np.sqrt(np.sum(np.abs(current)**2))
    return float(numerator / max(denominator, np.finfo(float).eps))


def snr_db(reference, estimate):
    reference, estimate = _real_array(reference), _real_array(estimate)
    if reference.shape != estimate.shape:
        raise ValueError("Referência e estimativa devem ter a mesma forma.")
    signal = np.sum(reference**2)
    error = np.sum((reference - estimate)**2)
    if signal == 0:
        raise ValueError("SNR indefinida para referência de energia zero.")
    return float(10 * np.log10(signal / error)) if error else float("inf")


def _reconstruction(d_hat, z_hat):
    return ifft2(np.sum(d_hat[..., None] * z_hat, axis=2))


def _objective(extended, z, image, shape, lambda_residual, beta, mask=None):
    error = crop_observed(extended, image.shape[:2], shape) - image
    if mask is not None:
        error = error * mask
    return float(0.5 * lambda_residual * np.sum(error**2) + beta * np.sum(np.abs(z)))


def _solve_coefficients(d_hat, xi_data_hat, xi_sparse_hat, rho):
    """Sherman–Morrison por frequência, sem construir matrizes K × K."""
    d = d_hat[..., None]
    rhs = d.conj() * xi_data_hat[:, :, None, :] + rho * xi_sparse_hat
    power = np.sum(np.abs(d)**2, axis=2, keepdims=True)
    correction = np.sum(d * rhs, axis=2, keepdims=True) / (rho + power)
    return (rhs - d.conj() * correction) / rho


def _dictionary_inverse(z_hat, rho):
    """Woodbury: inversa por frequência do sistema de atualização de D.

    A tem forma (..., N, K). O sistema amortecido N × N é positivo definido;
    solve substitui a pseudoinversa do MATLAB sem alterar a solução exata.
    """
    a = z_hat.swapaxes(-1, -2)
    ah = a.conj().swapaxes(-1, -2)
    small = rho * np.eye(a.shape[-2]) + a @ ah
    return (np.eye(a.shape[-1]) - ah @ np.linalg.solve(small, a)) / rho


@dataclass
class ReconstructionResult:
    coefficients: np.ndarray
    reconstruction: np.ndarray
    history: list[dict]
    objective: float
    iterations: int
    stop_reason: str
    elapsed_seconds: float


@dataclass
class DictionaryResult:
    filters: np.ndarray
    full_filters: np.ndarray
    coefficients: np.ndarray
    reconstruction: np.ndarray
    reconstruction_extended: np.ndarray
    filter_history: np.ndarray
    history: list[dict]
    objective: float
    iterations: int
    attempted_iterations: int
    stop_reason: str
    elapsed_seconds: float


def sparse_reconstruct(image, filters, *, beta=0.5, lambda_residual=1.0, max_iter=80, tol=1e-3, mask=None):
    """ADMM com filtros fixos, equivalente ao solver de reconstrução MATLAB.

    image é 2D; mask, quando fornecida, deve ser binária e da mesma forma.
    A saída conserva o iterado z do MATLAB, sem substituí-lo pelo proximal.
    """
    image, filters = _real_array(image, 2), _real_array(filters, 3)
    shape = _odd_shape(filters.shape[:2])
    _parameters(image, beta, lambda_residual, max_iter, tol)
    if mask is None:
        mask = np.ones_like(image)
    mask = _real_array(mask, 2)
    if mask.shape != image.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("mask deve ser binária e ter a mesma forma da imagem.")
    start = perf_counter()
    radius = (shape[0] // 2, shape[1] // 2)
    spatial = (image.shape[0] + shape[0] - 1, image.shape[1] + shape[1] - 1)
    pad = ((radius[0], radius[0]), (radius[1], radius[1]))
    m = np.pad(mask, pad)[..., None]
    mtb = np.pad(image * mask, pad)[..., None]
    d_hat = fft2(embed_filters(filters, spatial))
    gamma = 60 * beta / image.max()
    gamma_data = gamma / 100
    z = np.zeros((*spatial, filters.shape[2], 1))
    z_hat = fft2(z)
    dual_data, dual_sparse = np.zeros((*spatial, 1)), np.zeros_like(z)
    extended = _reconstruction(d_hat, z_hat)
    history = [{"iteration": 0, "objective": _objective(extended, z, image[..., None], shape, lambda_residual, beta, mask[..., None]), "relative_change": 0.0, "elapsed_seconds": 0.0}]
    stop_reason = "max_iter"
    for i in range(1, int(max_iter) + 1):
        v_data = extended
        theta = lambda_residual / gamma_data
        u_data = (mtb + (v_data - dual_data) / theta) / (m + 1 / theta)
        u_sparse = soft_threshold(z - dual_sparse, beta / gamma)
        dual_data -= v_data - u_data
        dual_sparse -= z - u_sparse
        xi_data_hat = fft2(u_data + dual_data)
        xi_sparse_hat = fft2(u_sparse + dual_sparse)
        old = z
        z_hat = _solve_coefficients(d_hat, xi_data_hat, xi_sparse_hat, gamma / gamma_data)
        z = ifft2(z_hat)
        extended = _reconstruction(d_hat, z_hat)
        change = relative_change(z, old)
        objective = _objective(extended, z, image[..., None], shape, lambda_residual, beta, mask[..., None])
        if not np.isfinite(objective):
            raise FloatingPointError("ADMM produziu objetivo não finito.")
        history.append({"iteration": i, "objective": objective, "relative_change": change, "elapsed_seconds": perf_counter() - start})
        if change < tol:
            stop_reason = "relative_change"
            break
    return ReconstructionResult(z[..., 0], crop_observed(extended, image.shape, shape)[..., 0], history, objective, i, stop_reason, perf_counter() - start)


def learn_dictionary(images, filter_shape=(21, 11), n_filters=3, *, beta=1.0, lambda_residual=1.0, max_iter=80, inner_iter=10, tol=1e-3, seed=2026, initial_state=None):
    """Alterna ADMM para D e Z, preservando a regra de restauração MATLAB.

    initial_state=(d_full, z) permite comparações com estados idênticos entre
    implementações. d_full já deve ter a origem deslocada, como embed_filters.
    A reconstrução de treinamento usa o filtro estendido retornado pelo ADMM;
    a etapa posterior de denoising usa os filtros compactos, como o MATLAB.
    """
    images = _real_array(images)
    if images.ndim not in (2, 3):
        raise ValueError("images deve ter forma (H, W) ou (H, W, N).")
    image = images[..., None] if images.ndim == 2 else images
    shape = _odd_shape(filter_shape)
    if int(n_filters) != n_filters or n_filters < 1 or int(inner_iter) != inner_iter or inner_iter < 1:
        raise ValueError("n_filters e inner_iter devem ser inteiros positivos.")
    n_filters, inner_iter = int(n_filters), int(inner_iter)
    _parameters(image, beta, lambda_residual, max_iter, tol)
    start = perf_counter()
    spatial = (image.shape[0] + shape[0] - 1, image.shape[1] + shape[1] - 1)
    radius = (shape[0] // 2, shape[1] // 2)
    pad = ((radius[0], radius[0]), (radius[1], radius[1]), (0, 0))
    m, mtb = np.pad(np.ones_like(image), pad), np.pad(image, pad)
    z_shape = (*spatial, n_filters, image.shape[2])
    if initial_state is None:
        rng = np.random.default_rng(seed)
        d = embed_filters(rng.standard_normal((*shape, n_filters)), spatial)
        z = rng.standard_normal(z_shape)
    else:
        d, z = (_real_array(a).copy() for a in initial_state)
        if d.shape != (*spatial, n_filters) or z.shape != z_shape:
            raise ValueError("initial_state tem dimensões incompatíveis com o problema.")
    d_hat, z_hat = fft2(d), fft2(z)
    dual_d_data, dual_d_filter = np.zeros_like(m), np.zeros_like(d)
    dual_z_data, dual_z_sparse = np.zeros_like(m), np.zeros_like(z)
    gamma = 60 * beta / image.max()
    gamma_d_data, gamma_z_data = gamma / 5000, gamma / 500
    extended = _reconstruction(d_hat, z_hat)
    objective = _objective(extended, z, image, shape, lambda_residual, beta)
    objective_d = objective_z = objective
    history = [{"iteration": 0, "objective_d": objective, "objective_z": objective, "relative_d": 0.0, "relative_z": 0.0, "elapsed_seconds": 0.0}]
    snapshots = [compact_filters(d, shape)]
    stop_reason = "max_iter"
    for i in range(1, int(max_iter) + 1):
        old_d, old_d_hat, old_z, old_z_hat = d, d_hat, z, z_hat
        old_min = min(objective_d, objective_z)
        rho_d = gamma / gamma_d_data
        inverse = _dictionary_inverse(z_hat, rho_d)
        for _ in range(inner_iter):
            v_data = _reconstruction(d_hat, z_hat)
            theta = lambda_residual / gamma_d_data
            u_data = (mtb + (v_data - dual_d_data) / theta) / (m + 1 / theta)
            u_filter = project_filters(d - dual_d_filter, shape)
            dual_d_data -= v_data - u_data
            dual_d_filter -= d - u_filter
            xi_data_hat, xi_filter_hat = fft2(u_data + dual_d_data), fft2(u_filter + dual_d_filter)
            rhs = np.sum(z_hat.conj() * xi_data_hat[:, :, None, :], axis=3) + rho_d * xi_filter_hat
            d_hat = np.einsum("...ij,...j->...i", inverse, rhs)
            d = ifft2(d_hat)
        objective_d = _objective(_reconstruction(d_hat, z_hat), z, image, shape, lambda_residual, beta)
        for _ in range(inner_iter):
            v_data = _reconstruction(d_hat, z_hat)
            theta = lambda_residual / gamma_z_data
            u_data = (mtb + (v_data - dual_z_data) / theta) / (m + 1 / theta)
            u_sparse = soft_threshold(z - dual_z_sparse, beta / gamma)
            dual_z_data -= v_data - u_data
            dual_z_sparse -= z - u_sparse
            z_hat = _solve_coefficients(d_hat, fft2(u_data + dual_z_data), fft2(u_sparse + dual_z_sparse), gamma / gamma_z_data)
            z = ifft2(z_hat)
        extended = _reconstruction(d_hat, z_hat)
        objective_z = _objective(extended, z, image, shape, lambda_residual, beta)
        if not np.isfinite([objective_d, objective_z]).all():
            raise FloatingPointError("Treinamento produziu objetivo não finito.")
        if old_min <= objective_d and old_min <= objective_z:
            d, d_hat, z, z_hat = old_d, old_d_hat, old_z, old_z_hat
            extended = _reconstruction(d_hat, z_hat)
            objective = _objective(extended, z, image, shape, lambda_residual, beta)
            stop_reason = "no_progress_rollback"
            break
        objective = objective_z
        change_d, change_z = relative_change(d, old_d), relative_change(z, old_z)
        history.append({"iteration": i, "objective_d": objective_d, "objective_z": objective_z, "relative_d": change_d, "relative_z": change_z, "elapsed_seconds": perf_counter() - start})
        snapshots.append(compact_filters(d, shape))
        if change_d < tol and change_z < tol:
            stop_reason = "relative_change"
            break
    return DictionaryResult(compact_filters(d, shape), d, z, crop_observed(extended, image.shape[:2], shape), extended, np.stack(snapshots, axis=3), history, objective, len(history) - 1, i, stop_reason, perf_counter() - start)
