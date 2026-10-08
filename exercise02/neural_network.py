"""Redes neurais densas em NumPy para os experimentos do Capítulo 4.

Esta é uma conversão dos códigos MATLAB ``NNode1.m``, ``NNode_test.m`` e
``NNode_multiclass.m``. A implementação mantém a convenção do material:
exemplos nas colunas, matrizes de pesos com o bias na primeira coluna e
treinamento em lote completo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass
class TrainingResult:
    """Resultado de um treinamento e informações de diagnóstico."""

    weights: list[Array]
    losses: Array
    step_sizes: Array
    success: bool = True
    message: str = ""


def sigmoid(z: Array) -> Array:
    """Sigmoide estável numericamente."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def activate(z: Array, kind: str) -> tuple[Array, Array | None]:
    """Retorna ativação e derivada elemento a elemento.

    O ``NNode_multiclass.m`` chama sua opção 2 de ReLU, mas o arquivo
    ``activation.m`` baixado do laboratório implementa ELU com alpha=1.
    Por isso as duas opções são oferecidas separadamente.
    """
    if kind == "sigmoid":
        value = sigmoid(z)
        return value, value * (1.0 - value)
    if kind == "relu":
        return np.maximum(z, 0.0), (z > 0.0).astype(np.float64)
    if kind == "elu":
        value = np.where(z >= 0.0, z, np.expm1(z))
        derivative = np.where(z >= 0.0, 1.0, np.exp(z))
        return value, derivative
    if kind == "softmax":
        shifted = z - np.max(z, axis=0, keepdims=True)
        exponential = np.exp(shifted)
        return exponential / exponential.sum(axis=0, keepdims=True), None
    raise ValueError(f"Ativação desconhecida: {kind}")


def generate_binary_data(
    n_examples: int = 100,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[Array, Array]:
    """Reproduz ``Datain1.m``: metade dos vetores é o vetor nulo."""
    if n_examples % 2:
        raise ValueError("n_examples deve ser par para balanceamento exato")
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_features, n_examples)).astype(np.float64)
    target = (x.sum(axis=0, keepdims=True) == 0).astype(np.float64)

    n_positive = int(target.sum())
    candidates = np.flatnonzero(target.ravel() == 0)
    needed = n_examples // 2 - n_positive
    x[:, candidates[:needed]] = 0.0
    target[:, candidates[:needed]] = 1.0

    permutation = rng.permutation(n_examples)
    return x[:, permutation], target[:, permutation]


def generate_multiclass_data(
    examples_per_class: int = 50,
    n_features: int = 5,
    seed: int = 84,
) -> tuple[Array, Array]:
    """Gera as seis classes de ``NNode_multiclass.m``.

    A classe é a quantidade de bits iguais a um no vetor de cinco posições.
    """
    rng = np.random.default_rng(seed)
    n_classes = n_features + 1
    n_examples = examples_per_class * n_classes
    x = np.empty((n_features, n_examples), dtype=np.float64)
    target = np.zeros((n_classes, n_examples), dtype=np.float64)

    column = 0
    for class_index in range(n_classes):
        base = np.r_[np.ones(class_index), np.zeros(n_features - class_index)]
        for _ in range(examples_per_class):
            x[:, column] = rng.permutation(base)
            target[class_index, column] = 1.0
            column += 1

    permutation = rng.permutation(n_examples)
    return x[:, permutation], target[:, permutation]


def initialize_weights(layer_sizes: Sequence[int], seed: int = 123) -> list[Array]:
    """Inicializa pesos uniformes em [0, 0.1), como os scripts MATLAB."""
    rng = np.random.default_rng(seed)
    return [
        rng.random((n_out, n_in + 1), dtype=np.float64) * 0.1
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def forward(
    x: Array,
    weights: Sequence[Array],
    hidden_activation: str,
    output_activation: str,
) -> tuple[Array, list[Array], list[Array], list[Array]]:
    """Propagação direta; devolve também o cache usado no backpropagation."""
    activations = [x]
    activations_with_bias: list[Array] = []
    preactivations: list[Array] = []

    current = x
    for layer_index, weight in enumerate(weights):
        biased = np.vstack((np.ones((1, current.shape[1])), current))
        z = weight @ biased
        kind = output_activation if layer_index == len(weights) - 1 else hidden_activation
        current, _ = activate(z, kind)
        activations_with_bias.append(biased)
        preactivations.append(z)
        activations.append(current)
    return current, activations, activations_with_bias, preactivations


def _loss_and_output_delta(
    prediction: Array,
    target: Array,
    output_activation: str,
    loss_name: str,
) -> tuple[float, Array]:
    n_examples = target.shape[1]
    if loss_name == "l2":
        residual = prediction - target
        loss = float(np.sum(residual**2) / n_examples)
        derivative_prediction = 2.0 * residual / n_examples
    elif loss_name == "binary_cross_entropy":
        clipped = np.clip(prediction, 1e-12, 1.0 - 1e-12)
        loss = float(
            np.sum(-target * np.log(clipped) - (1.0 - target) * np.log1p(-clipped))
            / n_examples
        )
        derivative_prediction = (
            -target / clipped + (1.0 - target) / (1.0 - clipped)
        ) / n_examples
    elif loss_name == "categorical_cross_entropy":
        clipped = np.clip(prediction, 1e-12, 1.0)
        loss = float(-np.sum(target * np.log(clipped)) / n_examples)
        if output_activation != "softmax":
            raise ValueError("Entropia cruzada categórica requer saída softmax")
        return loss, (prediction - target) / n_examples
    else:
        raise ValueError(f"Perda desconhecida: {loss_name}")

    if output_activation == "sigmoid":
        return loss, derivative_prediction * prediction * (1.0 - prediction)
    if output_activation == "softmax":
        # Produto J_softmax @ dL/dp, feito por coluna sem formar o tensor 3-D.
        projection = np.sum(derivative_prediction * prediction, axis=0, keepdims=True)
        return loss, prediction * (derivative_prediction - projection)
    raise ValueError(f"Ativação de saída desconhecida: {output_activation}")


def loss_and_gradients(
    x: Array,
    target: Array,
    weights: Sequence[Array],
    hidden_activation: str = "sigmoid",
    output_activation: str = "sigmoid",
    loss_name: str = "l2",
    regularization: float = 0.0,
) -> tuple[float, list[Array]]:
    """Calcula a perda média e seu gradiente analítico por backpropagation."""
    prediction, _, biased_activations, preactivations = forward(
        x, weights, hidden_activation, output_activation
    )
    loss, delta = _loss_and_output_delta(
        prediction, target, output_activation, loss_name
    )
    gradients: list[Array] = [np.empty_like(weight) for weight in weights]

    for layer_index in range(len(weights) - 1, -1, -1):
        gradients[layer_index] = delta @ biased_activations[layer_index].T
        if regularization:
            # O bias não é regularizado, conforme a observação da seção 4.3.1.
            loss += regularization * float(np.sum(weights[layer_index][:, 1:] ** 2)) / (
                2.0 * x.shape[1]
            )
            gradients[layer_index][:, 1:] += (
                regularization / x.shape[1]
            ) * weights[layer_index][:, 1:]
        if layer_index:
            propagated = weights[layer_index][:, 1:].T @ delta
            _, derivative = activate(
                preactivations[layer_index - 1], hidden_activation
            )
            delta = propagated * derivative
    return loss, gradients


def predict(
    x: Array,
    weights: Sequence[Array],
    hidden_activation: str,
    output_activation: str,
) -> Array:
    """Calcula probabilidades/saídas da rede."""
    return forward(x, weights, hidden_activation, output_activation)[0]


def binary_accuracy(target: Array, prediction: Array, threshold: float = 0.34) -> float:
    return float(np.mean((prediction >= threshold) == target))


def multiclass_accuracy(target: Array, prediction: Array) -> float:
    return float(np.mean(np.argmax(prediction, axis=0) == np.argmax(target, axis=0)))


def train_backtracking(
    x: Array,
    target: Array,
    weights: Sequence[Array],
    hidden_activation: str,
    output_activation: str,
    loss_name: str,
    n_iterations: int = 400,
    initial_step: float = 1.0,
    minimum_step: float = 1e-5,
) -> TrainingResult:
    """Descida mais íngreme com busca de passo por bisseção."""
    current = [weight.copy() for weight in weights]
    losses = np.empty(n_iterations, dtype=np.float64)
    steps = np.empty(n_iterations, dtype=np.float64)

    for iteration in range(n_iterations):
        loss, gradients = loss_and_gradients(
            x, target, current, hidden_activation, output_activation, loss_name
        )
        losses[iteration] = loss
        step = initial_step
        while True:
            candidate = [
                weight - step * gradient
                for weight, gradient in zip(current, gradients)
            ]
            candidate_loss, _ = loss_and_gradients(
                x, target, candidate, hidden_activation, output_activation, loss_name
            )
            if candidate_loss <= loss or step <= minimum_step:
                break
            step *= 0.5
        if candidate_loss <= loss:
            current = candidate
        else:
            step = 0.0
        steps[iteration] = step

    return TrainingResult(current, losses, steps)


def pack_weights(weights: Sequence[Array]) -> tuple[Array, list[tuple[int, int]]]:
    shapes = [weight.shape for weight in weights]
    return np.concatenate([weight.ravel() for weight in weights]), shapes


def unpack_weights(parameters: Array, shapes: Iterable[tuple[int, int]]) -> list[Array]:
    weights: list[Array] = []
    start = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(parameters[start : start + size].reshape(shape))
        start += size
    return weights


def train_conjugate_gradient(
    x: Array,
    target: Array,
    weights: Sequence[Array],
    hidden_activation: str = "sigmoid",
    loss_name: str = "l2",
    n_iterations: int = 200,
) -> TrainingResult:
    """Equivalente NumPy/SciPy do ``fmincg`` usado no MATLAB multiclasse."""
    from scipy.optimize import minimize

    initial, shapes = pack_weights(weights)
    history: list[float] = []

    def objective(parameters: Array) -> tuple[float, Array]:
        unpacked = unpack_weights(parameters, shapes)
        loss, gradients = loss_and_gradients(
            x,
            target,
            unpacked,
            hidden_activation=hidden_activation,
            output_activation="softmax",
            loss_name=loss_name,
        )
        gradient, _ = pack_weights(gradients)
        return loss, gradient

    def callback(parameters: Array) -> None:
        history.append(objective(parameters)[0])

    optimized = minimize(
        objective,
        initial,
        jac=True,
        method="CG",
        callback=callback,
        options={"maxiter": n_iterations, "gtol": 1e-8},
    )
    if not history or history[-1] != float(optimized.fun):
        history.append(float(optimized.fun))
    return TrainingResult(
        unpack_weights(optimized.x, shapes),
        np.asarray(history),
        np.full(len(history), np.nan),
        success=bool(optimized.success),
        message=str(optimized.message),
    )


def finite_difference_gradients(
    x: Array,
    target: Array,
    weights: Sequence[Array],
    hidden_activation: str,
    output_activation: str,
    loss_name: str,
    epsilon: float = 1e-4,
) -> list[Array]:
    """Gradiente por diferenças centrais, como em ``NNode_test.m``."""
    numerical = [np.zeros_like(weight) for weight in weights]
    for layer_index, weight in enumerate(weights):
        for index in np.ndindex(weight.shape):
            lower = [item.copy() for item in weights]
            upper = [item.copy() for item in weights]
            lower[layer_index][index] -= epsilon
            upper[layer_index][index] += epsilon
            lower_loss, _ = loss_and_gradients(
                x, target, lower, hidden_activation, output_activation, loss_name
            )
            upper_loss, _ = loss_and_gradients(
                x, target, upper, hidden_activation, output_activation, loss_name
            )
            numerical[layer_index][index] = (upper_loss - lower_loss) / (2.0 * epsilon)
    return numerical


def gradient_check(
    x: Array,
    target: Array,
    weights: Sequence[Array],
    hidden_activation: str = "relu",
    output_activation: str = "sigmoid",
    loss_name: str = "binary_cross_entropy",
    epsilon: float = 1e-4,
) -> dict[str, object]:
    """Compara gradientes analíticos e numéricos camada a camada."""
    _, analytical = loss_and_gradients(
        x, target, weights, hidden_activation, output_activation, loss_name
    )
    numerical = finite_difference_gradients(
        x,
        target,
        weights,
        hidden_activation,
        output_activation,
        loss_name,
        epsilon,
    )
    differences = [a - n for a, n in zip(analytical, numerical)]
    normalized = [
        difference / max(float(np.max(np.abs(gradient))), np.finfo(float).eps)
        for difference, gradient in zip(differences, analytical)
    ]
    flat_analytical = np.concatenate([item.ravel() for item in analytical])
    flat_numerical = np.concatenate([item.ravel() for item in numerical])
    relative_l2 = float(
        np.linalg.norm(flat_analytical - flat_numerical)
        / max(
            np.linalg.norm(flat_analytical) + np.linalg.norm(flat_numerical),
            np.finfo(float).eps,
        )
    )
    return {
        "analytical": analytical,
        "numerical": numerical,
        "normalized_differences": normalized,
        "max_absolute_error": float(
            max(np.max(np.abs(item)) for item in differences)
        ),
        "relative_l2_error": relative_l2,
    }
