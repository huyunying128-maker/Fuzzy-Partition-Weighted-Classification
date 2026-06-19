"""Polynomial feature maps for local classification models."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


@dataclass(frozen=True)
class PolynomialBasisSpec:
    """Description of a polynomial basis."""

    degree: int
    include_bias: bool
    interaction_mode: str
    n_input_features: int
    n_output_features: int


def _as_2d_float_array(array: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a finite two-dimensional float array."""
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if result.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _check_degree(degree: int) -> int:
    """Validate a nonnegative polynomial degree."""
    value = int(degree)
    if value < 0:
        raise ValueError("degree must be nonnegative.")
    return value


def _check_interaction_mode(interaction_mode: str) -> str:
    """Validate the polynomial interaction mode."""
    key = str(interaction_mode).lower().replace("-", "_").strip()
    if key not in {"full", "powers"}:
        raise ValueError("interaction_mode must be either 'full' or 'powers'.")
    return key


def polynomial_feature_count(
    n_features: int,
    degree: int,
    include_bias: bool = True,
    interaction_mode: str = "full",
) -> int:
    """Return the number of output features in a polynomial basis."""
    p = int(n_features)
    if p <= 0:
        raise ValueError("n_features must be positive.")
    q = _check_degree(degree)
    mode = _check_interaction_mode(interaction_mode)

    if mode == "full":
        total = comb(p + q, q)
        if not include_bias:
            total -= 1
        return int(total)

    total = p * q
    if include_bias:
        total += 1
    return int(total)


def polynomial_basis(
    x: np.ndarray,
    degree: int,
    include_bias: bool = True,
    interaction_mode: str = "full",
) -> np.ndarray:
    """Construct polynomial features for an input matrix."""
    transformer = PolynomialBasis(
        degree=degree,
        include_bias=include_bias,
        interaction_mode=interaction_mode,
    )
    return transformer.fit_transform(x)


class PolynomialBasis:
    """Polynomial feature transformer.

    Parameters
    ----------
    degree:
        Maximum polynomial degree.
    include_bias:
        Include a leading constant column.
    interaction_mode:
        ``"full"`` includes all monomials up to the selected degree. ``"powers"``
        includes only coordinate-wise powers.
    """

    def __init__(
        self,
        degree: int = 1,
        include_bias: bool = True,
        interaction_mode: str = "full",
    ) -> None:
        self.degree = _check_degree(degree)
        self.include_bias = bool(include_bias)
        self.interaction_mode = _check_interaction_mode(interaction_mode)
        self._sklearn_transformer: PolynomialFeatures | None = None
        self.n_features_in_: int | None = None
        self.n_output_features_: int | None = None
        self.spec_: PolynomialBasisSpec | None = None

    def fit(self, x: np.ndarray) -> "PolynomialBasis":
        """Fit the feature transformer to an input matrix."""
        x_array = _as_2d_float_array(x, "x")
        self.n_features_in_ = int(x_array.shape[1])

        if self.interaction_mode == "full":
            transformer = PolynomialFeatures(
                degree=self.degree,
                include_bias=self.include_bias,
                interaction_only=False,
                order="C",
            )
            transformer.fit(x_array)
            self._sklearn_transformer = transformer
            self.n_output_features_ = int(transformer.n_output_features_)
        else:
            self._sklearn_transformer = None
            self.n_output_features_ = polynomial_feature_count(
                n_features=self.n_features_in_,
                degree=self.degree,
                include_bias=self.include_bias,
                interaction_mode="powers",
            )

        self.spec_ = PolynomialBasisSpec(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_mode=self.interaction_mode,
            n_input_features=self.n_features_in_,
            n_output_features=self.n_output_features_,
        )
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform an input matrix into polynomial features."""
        if self.n_features_in_ is None:
            raise RuntimeError("PolynomialBasis must be fitted before transform.")

        x_array = _as_2d_float_array(x, "x")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(
                "x has a different number of features from the fitted input: "
                f"expected {self.n_features_in_}, got {x_array.shape[1]}."
            )

        if self.interaction_mode == "full":
            if self._sklearn_transformer is None:
                raise RuntimeError("internal polynomial transformer has not been fitted.")
            return self._sklearn_transformer.transform(x_array).astype(np.float64, copy=False)

        blocks: list[np.ndarray] = []
        if self.include_bias:
            blocks.append(np.ones((x_array.shape[0], 1), dtype=np.float64))
        for power in range(1, self.degree + 1):
            blocks.append(x_array**power)
        if not blocks:
            raise ValueError("the requested basis contains no output features.")
        return np.concatenate(blocks, axis=1)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the transformer and return polynomial features."""
        return self.fit(x).transform(x)

    def get_feature_names(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names after fitting."""
        if self.n_features_in_ is None:
            raise RuntimeError("PolynomialBasis must be fitted before feature names are available.")

        if input_features is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    "input_features must have length "
                    f"{self.n_features_in_}, got {len(input_features)}."
                )
            names = list(input_features)

        if self.interaction_mode == "full":
            if self._sklearn_transformer is None:
                raise RuntimeError("internal polynomial transformer has not been fitted.")
            return list(self._sklearn_transformer.get_feature_names_out(names))

        output_names: list[str] = []
        if self.include_bias:
            output_names.append("1")
        for power in range(1, self.degree + 1):
            if power == 1:
                output_names.extend(names)
            else:
                output_names.extend([f"{name}^{power}" for name in names])
        return output_names
