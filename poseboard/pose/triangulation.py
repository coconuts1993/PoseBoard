"""多相机关键点三角化（加权 DLT）。"""

from __future__ import annotations

import numpy as np

from poseboard.calibration import CameraCalibration


def triangulate_keypoints(cams: list[CameraCalibration], poses_2d: list[np.ndarray],
                          scores: list[np.ndarray], min_score: float = 0.5
                          ) -> tuple[np.ndarray, np.ndarray]:
    """对每个关键点做加权 DLT。

    poses_2d[i]: (K,2) 第 i 台相机的像素坐标；scores[i]: (K,) 置信度。
    返回 (K,3) 世界坐标（不足两视角的点为 NaN）以及 (K,) 平均置信度。
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
