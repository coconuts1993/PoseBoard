"""Temporal smoothing of keypoint trajectories.

``OneEuroFilter`` (Casiez, Roussel and Vogel, CHI 2012): a low-pass filter whose cutoff
frequency rises with the speed of the signal, so a still keypoint is smoothed (less jitter)
while fast movements are followed with little lag. It works element-wise on arrays, e.g. the
(K, 3) keypoints of a pose; an element that becomes NaN (a keypoint that disappeared) is reset
and starts afresh when it comes back.

The defaults are chosen for keypoint coordinates in meters at 15-30 poses per second:
``min_cutoff = 2 Hz`` keeps about 90 % of the amplitude of a 1 Hz sway (97 % at 0.5 Hz) and
roughly halves the jitter of a still keypoint; ``beta = 4 s/m`` raises the cutoff with the
speed, so a movement of about 1.5 m/s lags by about one frame. (``beta`` depends on the units
of the signal: a value tuned for pixels, such as 0.01, would switch the speed adaptation off
for meters.)
"""

from __future__ import annotations

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Element-wise One-Euro filter.

    ``min_cutoff`` (Hz): cutoff at zero speed (lower = smoother, more lag at rest);
    ``beta`` (s per unit of the signal, e.g. s/m): how fast the cutoff rises with speed (higher
    = less lag when moving; the cutoff is ``min_cutoff + beta * speed``); ``d_cutoff`` (Hz):
    cutoff of the speed estimate. The defaults suit coordinates in meters (see the module
    docstring). Call ``f(x, t)`` with increasing times ``t`` (seconds)."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 4.0, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("OneEuroFilter needs min_cutoff > 0, d_cutoff > 0 and beta >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None  # filtered value (NaN: element not initialised)
        self._dx: np.ndarray | None = None  # filtered derivative
        self._t: float | None = None

    def __call__(self, x, t: float):
        scalar = np.ndim(x) == 0
        x = np.array(x, np.float64, copy=True)
        t = float(t)
        if self._x is None or self._x.shape != x.shape or self._t is None or t < self._t:
            self._x = x.copy()
            self._dx = np.where(np.isfinite(x), 0.0, np.nan)
            self._t = t
            return float(x) if scalar else x
        dt = t - self._t
        valid = np.isfinite(x)
        known = valid & np.isfinite(self._x)
        out = x.copy()
        if dt <= 0:  # same instant again: keep the current estimate
            out[known] = self._x[known]
        else:
            dx = np.zeros_like(x)
            dx[known] = (x[known] - self._x[known]) / dt
            a_d = _alpha(self.d_cutoff, dt)
            dx_hat = np.where(known, a_d * dx + (1 - a_d) * np.nan_to_num(self._dx), 0.0)
            cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
            a = _alpha(cutoff, dt)
            out[known] = a[known] * x[known] + (1 - a[known]) * self._x[known]
            self._dx = np.where(valid, dx_hat, np.nan)
            self._t = t
        # new elements start from their raw value; vanished elements are reset (NaN)
        self._x = np.where(valid, out, np.nan)
        if dt <= 0:
            self._dx = np.where(valid, np.nan_to_num(self._dx), np.nan)
        return float(out) if scalar else out
