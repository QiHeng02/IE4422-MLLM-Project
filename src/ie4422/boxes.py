"""Box formats, Qwen2.5-VL resizing and IoU matching."""
from __future__ import annotations

import math

QWEN_FACTOR = 28  # Qwen2.5-VL patch (14) x spatial merge (2)


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    """YOLO normalised (centre, size) -> pixel corners [x1, y1, x2, y2], clipped to the image."""
    x1 = max(0.0, (cx - w / 2) * img_w)
    y1 = max(0.0, (cy - h / 2) * img_h)
    x2 = min(float(img_w), (cx + w / 2) * img_w)
    y2 = min(float(img_h), (cy + h / 2) * img_h)
    return [x1, y1, x2, y2]


def smart_resize(height: int, width: int, min_pixels: int, max_pixels: int,
                 factor: int = QWEN_FACTOR) -> tuple[int, int]:
    """Reproduce Qwen2.5-VL's image resize, so answer boxes match what the model sees.

    Qwen2.5-VL grounds boxes in absolute pixels of the RESIZED image, not the original.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme for Qwen2.5-VL")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def scale_box(box: list[float], src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> list[int]:
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return [round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy)]


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds: list[list[float]], gts: list[list[float]], thr: float) -> int:
    """Number of ground-truth boxes matched one-to-one by a prediction at IoU >= thr."""
    pairs = sorted(
        ((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
        reverse=True,
    )
    used_p, used_g, matched = set(), set(), 0
    for score, i, j in pairs:
        if score < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched += 1
    return matched
