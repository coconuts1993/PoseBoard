"""Multi-camera keypoint triangulation (weighted DLT, optionally with outlier rejection)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poseboard.calibration import CameraCalibration


def triangulate_keypoints(cams: list[CameraCalibration], poses_2d: list[np.ndarray],
                          scores: list[np.ndarray], min_score: float = 0.5
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Weighted DLT for each keypoint.

    poses_2d[i]: (K,2) pixel coordinates from camera i; scores[i]: (K,) confidences.
    Returns (K,3) world coordinates (NaN for points seen in fewer than two views) and
    (K,) mean confidences.
    """
    K = poses_2d[0].shape[0]
    norm = [c.undistort_normalized(p) for c, p in zip(cams, poses_2d)]
    Ps = [c.projection_matrix(normalized=True) for c in cams]
    out = np.full((K, 3), np.nan)
    conf = np.zeros(K)
    for k in range(K):
        rows, ws = [], []
        for n, P, s in zip(norm, Ps, scores):
            w = float(s[k])
            if w < min_score or not np.all(np.isfinite(n[k])):
                continue
            x, y = n[k]
            rows.append(w * (x * P[2] - P[0]))
            rows.append(w * (y * P[2] - P[1]))
            ws.append(w)
        if len(ws) < 2:
            continue
        _, _, Vt = np.linalg.svd(np.asarray(rows))
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            continue
        out[k] = X[:3] / X[3]
        conf[k] = float(np.mean(ws))
    return out, conf


@dataclass
class Triangulation:
    """Result of ``triangulate_robust``."""

    points: np.ndarray  # (K, 3) world coordinates; NaN where fewer than 2 views remain
    scores: np.ndarray  # (K,) mean confidence of the views used (0 where NaN)
    used: np.ndarray  # (V, K) bool: view v was used for keypoint k
    errors: np.ndarray  # (V, K) reprojection error (px) of the final point in every view; NaN if unknown
    valid: np.ndarray  # (V, K) bool: the 2D keypoint was usable (finite, score >= min_score)

    @property
    def rejected(self) -> np.ndarray:
        """(V, K) bool: usable 2D keypoints dropped as outliers."""
        return self.valid & ~self.used & np.all(np.isfinite(self.points), axis=1)[None, :]

    @property
    def mean_error(self) -> float | None:
        """Mean reprojection error (px) over the (view, keypoint) pairs used; None if none."""
        e = self.errors[self.used & np.isfinite(self.errors)]
        return float(e.mean()) if e.size else None


def _dlt(norm: np.ndarray, Ps: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted DLT for all keypoints at once. norm (V, K, 2) normalized image coordinates
    (finite), Ps (V, 3, 4) normalized projection matrices, w (V, K) weights (0 = not used).
    Returns (K, 3), NaN where fewer than two views have a weight."""
    x, y = norm[..., 0:1], norm[..., 1:2]  # (V, K, 1)
    r1 = w[..., None] * (x * Ps[:, None, 2, :] - Ps[:, None, 0, :])  # (V, K, 4)
    r2 = w[..., None] * (y * Ps[:, None, 2, :] - Ps[:, None, 1, :])
    A = np.concatenate([r1, r2], axis=0).transpose(1, 0, 2)  # (K, 2V, 4)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]
    out = np.full((X.shape[0], 3), np.nan)
    ok = ((w > 0).sum(axis=0) >= 2) & (np.abs(X[:, 3]) > 1e-12)
    out[ok] = X[ok, :3] / X[ok, 3:4]
    return out


def reprojection_errors(cams: list[CameraCalibration], points: np.ndarray,
                        poses_2d: list[np.ndarray]) -> np.ndarray:
    """(V, K) pixel distance between each camera's 2D keypoints and the projection of the 3D
    points (inf for a point behind the camera, NaN where either is missing)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.full((len(cams), len(pts)), np.nan)
    ok = np.all(np.isfinite(pts), axis=1)
    if not ok.any():
        return out
    for v, (cam, p2) in enumerate(zip(cams, poses_2d)):
        proj = cam.project(pts[ok])
        depth = cam.world_to_camera(pts[ok])[:, 2]
        e = np.linalg.norm(proj - np.asarray(p2, np.float64)[ok], axis=1)
        e[depth <= 1e-6] = np.inf
        out[v, ok] = e
    return out


def triangulate_robust(cams: list[CameraCalibration], poses_2d: list[np.ndarray],
                       scores: list[np.ndarray], min_score: float = 0.3,
                       reproj_threshold_px: float = 15.0, min_views: int = 2) -> Triangulation:
    """Weighted DLT with iterative outlier rejection, per keypoint: while the reprojection
    error of the current point exceeds ``reproj_threshold_px`` in some view and at least
    ``min_views`` (>= 2) views would remain, drop the view that disagrees most with the others
    and triangulate again.

    The view to drop is the one whose removal leaves the smallest reprojection error in the
    remaining views (the outlier against the consensus of the others). The plain error of the
    all-view point is not used for this choice: a gross outlier pulls that point towards
    itself, so a good view can show the largest error.

    poses_2d[v]: (K, 2) pixels of camera v (NaN if missing); scores[v]: (K,) confidences.
    A keypoint is used from a view when its score >= ``min_score``; the DLT rows are weighted
    by the score."""
    V = len(cams)
    p2 = np.stack([np.asarray(p, np.float64).reshape(-1, 2) for p in poses_2d])  # (V, K, 2)
    sc = np.stack([np.asarray(s, np.float64).reshape(-1) for s in scores])  # (V, K)
    K = p2.shape[1]
    valid = np.all(np.isfinite(p2), axis=2) & np.isfinite(sc) & (sc >= min_score)
    w_all = np.where(valid, sc, 0.0)
    norm = np.zeros((V, K, 2))
    for v, cam in enumerate(cams):
        if valid[v].any():
            norm[v, valid[v]] = cam.undistort_normalized(p2[v, valid[v]])
    Ps = np.stack([c.projection_matrix(normalized=True) for c in cams])
    used = valid.copy()
    min_views = max(2, int(min_views))
    cols = np.arange(K)
    for _ in range(V + 1):
        pts = _dlt(norm, Ps, np.where(used, w_all, 0.0))
        err = reprojection_errors(cams, pts, list(p2))
        e_now = np.where(used & ~np.isnan(err), err, -np.inf).max(axis=0)
        cand = ((e_now > reproj_threshold_px) & (used.sum(axis=0) - 1 >= min_views)
                & np.all(np.isfinite(pts), axis=1))
        if not cand.any():
            break
        ci = cols[cand]
        # rest[v]: largest error of the other views when view v is left out (inf: not possible)
        rest = np.full((V, len(ci)), np.inf)
        for v in range(V):
            if not used[v, ci].any():
                continue
            w = np.where(used[:, ci], w_all[:, ci], 0.0)
            w[v] = 0.0
            pv = _dlt(norm[:, ci], Ps, w)
            ev = reprojection_errors(cams, pv, [p[ci] for p in p2])  # (V, n)
            others = used[:, ci].copy()
            others[v] = False
            ev = np.where(others & ~np.isnan(ev), ev, -np.inf).max(axis=0)
            ok = used[v, ci] & np.all(np.isfinite(pv), axis=1)
            rest[v] = np.where(ok, ev, np.inf)
        best = np.argmin(rest, axis=0)
        possible = np.isfinite(rest[best, np.arange(len(ci))])
        if not possible.any():
            break
        used[best[possible], ci[possible]] = False
    ok = np.all(np.isfinite(pts), axis=1)
    n = used.sum(axis=0)
    conf = np.where(ok & (n > 0), np.where(used, w_all, 0.0).sum(axis=0) / np.maximum(n, 1), 0.0)
    return Triangulation(pts, conf, used & ok[None, :], err, valid)
