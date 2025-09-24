#!/usr/bin/env python3
"""Gaussian classifier demo inspired by the MATLAB script in `lab02.m`.

This script generates a 1D Gaussian mixture dataset, fits Gaussian
classifiers with different covariance assumptions, evaluates them, and
produces plots for likelihoods, posteriors, and ROC curves (for the binary
case). The implementation mirrors the functionality of the MATLAB version
using Python libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc


# -----------------------------------------------------------------------------
# Data structures


@dataclass
class GaussianClassifier:
    """Container for Gaussian classifier parameters."""

    gtype: str
    priors: np.ndarray  # shape (K,)
    means: np.ndarray  # shape (K, D)
    variances: np.ndarray  # shape (K,) for 'I', shape (1,) for 'i'


# -----------------------------------------------------------------------------
# Utility functions


def gm_sample(
    n_samples: int,
    priors: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from a 1D (or D-dimensional isotropic) Gaussian mixture."""

    priors = np.asarray(priors, dtype=float)
    priors /= priors.sum()
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)

    rng = np.random.default_rng(random_state)
    n_components = priors.size
    components = rng.choice(n_components, size=n_samples, p=priors)

    X = np.empty((n_samples, means.shape[1]), dtype=float)
    for k in range(n_components):
        idx = components == k
        if not np.any(idx):
            continue
        sigma = math.sqrt(float(variances[k]))
        X[idx] = rng.normal(loc=means[k], scale=sigma, size=(idx.sum(), means.shape[1]))

    return X, components


def fit_gaussian_classifier(X: np.ndarray, y: np.ndarray, gtype: str) -> GaussianClassifier:
    """Estimate Gaussian classifier parameters under a given covariance type."""

    if gtype not in {"I", "i"}:
        raise ValueError(f"Unsupported covariance type '{gtype}'. Use 'I' or 'i'.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    K = classes.size
    D = X.shape[1]

    priors = np.zeros(K, dtype=float)
    means = np.zeros((K, D), dtype=float)
    variances = np.zeros(K if gtype == "I" else 1, dtype=float)

    for idx, k in enumerate(classes):
        Xk = X[y == k]
        priors[idx] = Xk.shape[0] / X.shape[0]
        means[idx] = Xk.mean(axis=0)
        sq_norm = np.sum((Xk - means[idx]) ** 2, axis=1)
        variances[idx if gtype == "I" else 0] += sq_norm.sum() / (Xk.shape[0] * D)

    if gtype == "I":
        variances = np.clip(variances, 1e-9, None)
    else:
        variances[:] = np.clip(variances[0] / K, 1e-9, None)

    return GaussianClassifier(gtype=gtype, priors=priors, means=means, variances=variances)


def gaussian_pdf(
    X: np.ndarray, model: GaussianClassifier
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute p(x), p(x|C), p(C|x), and p(x, C) for all classes."""

    X = np.asarray(X, dtype=float)
    K = model.priors.size
    D = X.shape[1]

    px_given_c = np.zeros((X.shape[0], K), dtype=float)
    for k in range(K):
        mean = model.means[k]
        if model.gtype == "i":
            sigma2 = float(model.variances[0])
        else:
            sigma2 = float(model.variances[k])

        coef = 1.0 / ((2.0 * math.pi * sigma2) ** (D / 2))
        diff = X - mean
        exponent = -0.5 * np.sum(diff * diff, axis=1) / sigma2
        px_given_c[:, k] = coef * np.exp(exponent)

    px = px_given_c @ model.priors
    px = np.clip(px, 1e-12, None)
    px_and_c = px_given_c * model.priors
    p_c_given_x = px_and_c / px[:, None]

    return px, px_given_c, p_c_given_x, px_and_c


def evaluate_classifier(
    X: np.ndarray, y: np.ndarray, model: GaussianClassifier
) -> Dict[str, np.ndarray | float]:
    """Predict labels, compute error statistics and confusion matrix."""

    _, _, posteriors, _ = gaussian_pdf(X, model)
    y_pred = posteriors.argmax(axis=1)
    error = float(np.mean(y_pred != y))

    labels = np.unique(y)
    cn_counts = confusion_matrix(y, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cn_norm = cn_counts / cn_counts.sum(axis=1, keepdims=True)
        cn_norm = np.nan_to_num(cn_norm)

    metrics = {
        "y_pred": y_pred,
        "error": error,
        "confusion": cn_counts,
        "confusion_norm": cn_norm,
        "posteriors": posteriors,
    }

    if labels.size == 2:
        fpr, tpr, _ = roc_curve(y, posteriors[:, 1])
        metrics.update({"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)})

    return metrics


def plot_classifier_results(
    X: np.ndarray,
    y: np.ndarray,
    model: GaussianClassifier,
    evaluation: Dict[str, np.ndarray | float],
    x_grid: np.ndarray,
    idx: int,
) -> None:
    """Plot density and posterior curves for a Gaussian classifier."""

    _, px_given_c_grid, posteriors_grid, px_and_c_grid = gaussian_pdf(x_grid, model)
    colors = plt.cm.get_cmap("tab10", model.priors.size)

    fig = plt.figure(idx, figsize=(10, 8))
    fig.clf()

    ax1 = fig.add_subplot(2, 1, 1)
    for k in range(model.priors.size):
        ax1.plot(
            x_grid[:, 0],
            px_given_c_grid[:, k],
            label=f"p(x|C={k})",
            color=colors(k),
        )
        ax1.plot(
            x_grid[:, 0],
            px_and_c_grid[:, k],
            linestyle="--",
            linewidth=2,
            label=f"p(x,C={k})",
            color=colors(k),
        )

    ax1.scatter(
        X[:, 0],
        np.zeros_like(X[:, 0]) - 0.05 * px_given_c_grid.max(),
        c=[colors(int(cls)) for cls in y],
        marker="x",
        s=70,
        label="samples",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("p(x|C), p(x,C)")
    ax1.set_title(
        f"1D dataset; covariance type: {model.gtype} | error = {evaluation['error'] * 100:.2f}%"
    )
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2, 1, 2)
    for k in range(model.priors.size):
        ax2.plot(
            x_grid[:, 0],
            posteriors_grid[:, k],
            linewidth=2,
            label=f"p(C={k}|x)",
            color=colors(k),
        )

    predicted_grid = posteriors_grid.argmax(axis=1)
    ax2.scatter(
        x_grid[:, 0],
        np.ones_like(x_grid[:, 0]) * 1.05,
        c=[colors(int(cls)) for cls in predicted_grid],
        s=50,
        marker=".",
        alpha=0.6,
        label="decision",
    )
    ax2.scatter(
        X[:, 0],
        np.zeros_like(X[:, 0]) - 0.05,
        c=[colors(int(cls)) for cls in y],
        marker="x",
        s=70,
    )
    ax2.set_ylim(-0.15, 1.1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("p(C|x)")
    ax2.legend(loc="upper right")

    fig.tight_layout()


def plot_roc_curves(roc_data: Dict[str, Dict[str, np.ndarray | float]]) -> None:
    """Plot ROC curves for all classifiers (binary problems only)."""

    if not roc_data:
        return

    plt.figure(figsize=(7, 6))
    plt.clf()
    for label, data in roc_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f"{label}; AUC={data['auc']:.3f}")
        plt.scatter(
            data["fp_rate"],
            data["tp_rate"],
            marker="*",
            s=120,
            label=f"{label} (operating point)",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("ROC curves on training set")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()


# -----------------------------------------------------------------------------
# Main experiment


def main() -> None:
    # Parameters mirroring the MATLAB script (1D, K=2)
    priors = np.array([2.0, 1.0])
    means = np.array([[0.0], [2.0]])
    variances = np.array([0.5, 3.0])
    priors /= priors.sum()

    n_samples = 100
    X, y = gm_sample(n_samples, priors, means, variances, random_state=37)

    # Grid for plotting densities
    x_min = float(np.min(means - 4 * np.sqrt(variances[:, None])))
    x_max = float(np.max(means + 4 * np.sqrt(variances[:, None])))
    x_grid = np.linspace(x_min, x_max, 1000, dtype=float)[:, None]

    classifiers = {}
    roc_info: Dict[str, Dict[str, np.ndarray | float]] = {}

    for idx, gtype in enumerate(["I", "i"], start=1):
        model = fit_gaussian_classifier(X, y, gtype)
        evaluation = evaluate_classifier(X, y, model)
        classifiers[gtype] = (model, evaluation)

        # Plot densities and posteriors
        plot_classifier_results(X, y, model, evaluation, x_grid, idx)

        if y.max() == 1:
            fpr = evaluation.get("fpr")
            tpr = evaluation.get("tpr")
            auc_value = evaluation.get("auc")
            cn_norm = evaluation["confusion_norm"]
            roc_info[gtype] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc_value,
                "fp_rate": cn_norm[0, 1],
                "tp_rate": cn_norm[1, 1],
            }

    plot_roc_curves(roc_info)

    plt.show()


if __name__ == "__main__":
    main()


