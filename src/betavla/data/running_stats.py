"""Running statistics for normalization (OpenPI-style).

Computes mean, std, q01, q99 per dimension via streaming batches.
Used by compute_norm_stats to align with OpenPI's normalization pipeline.
"""

from __future__ import annotations

import numpy as np


class RunningStats:
    """Compute running statistics of a batch of vectors (OpenPI-style)."""

    def __init__(self, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = num_quantile_bins

    @property
    def count(self) -> int:
        return self._count

    def update(self, batch: np.ndarray) -> None:
        """Update statistics with a batch. Last dim is the vector dimension."""
        batch = np.asarray(batch, dtype=np.float64)
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [
                np.zeros(self._num_quantile_bins, dtype=np.float64) for _ in range(vector_length)
            ]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError(
                    f"Vector length {vector_length} != initialized {self._mean.size}"
                )
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, list[float]]:
        """Return mean, std, q01, q99 as lists (per-dimension)."""
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        stddev = np.where(stddev < 1e-6, 1.0, stddev)

        q01, q99 = self._compute_quantiles([0.01, 0.99])

        return {
            "mean": self._mean.astype(np.float64).tolist(),
            "std": stddev.astype(np.float64).tolist(),
            "q01": q01.astype(np.float64).tolist(),
            "q99": q99.astype(np.float64).tolist(),
        }

    def _adjust_histograms(self) -> None:
        """Adjust histograms when min or max changes."""
        assert self._histograms is not None and self._bin_edges is not None
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(
                self._min[i], self._max[i], self._num_quantile_bins + 1
            )
            new_hist, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[i]
            )
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        assert self._histograms is not None and self._bin_edges is not None
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles: list[float]) -> list[np.ndarray]:
        """Compute quantiles from histograms."""
        assert self._histograms is not None and self._bin_edges is not None
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                idx = min(idx, len(edges) - 2)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results
