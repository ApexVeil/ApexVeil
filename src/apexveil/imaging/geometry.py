"""Robust planar geometry, separate from visible segmentation."""

import cv2
import numpy as np


def order_quad(q):
    """Return four unique convex corners, clockwise from upper-left."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    if not np.isfinite(q).all() or len(np.unique(q, axis=0)) != 4:
        raise ValueError("A quadrilateral requires four finite, distinct corners")
    center = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - center[1], q[:, 0] - center[0]))]
    q = np.roll(q, -np.argmin(q.sum(axis=1)), axis=0)
    if not cv2.isContourConvex(q) or cv2.contourArea(q) < 4:
        raise ValueError("Degenerate surface quadrilateral")
    return q


def mask_quad(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("Visible mask is empty")
    contour = np.concatenate(contours)
    hull = cv2.convexHull(contour)
    for eps in np.linspace(0.003, 0.08, 100):
        p = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(p) == 4:
            return order_quad(p), "convex_hull_four_corner_fit"
    return order_quad(
        cv2.boxPoints(cv2.minAreaRect(contour))
    ), "min_area_rectangle_fallback"


def fit_quad(mask):
    initial, _ = mask_quad(mask)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate(cs).reshape(-1, 2).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = initial[i], initial[(i + 1) % 4]
        edge = b - a
        length = np.linalg.norm(edge)
        d = edge / length
        t = (pts - a) @ d / length
        dist = np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0])
        chosen = pts[(t > 0.12) & (t < 0.88) & (dist < max(3, length * 0.08))]
        if len(chosen) < 6:
            return initial, "initial_quad_fallback"
        vx, vy, x, y = cv2.fitLine(chosen, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        point = np.array([x, y])
        direction = np.array([vx, vy])
        normal = np.array([-vy, vx])
        if np.dot(initial.mean(0) - point, normal) < 0:
            normal = -normal
        lines.append((point - normal * 1.0, direction))
    corners = []
    for i in range(4):
        p, v = lines[(i - 1) % 4]
        q, u = lines[i]
        mat = np.column_stack((v, -u))
        if abs(np.linalg.det(mat)) < 0.05:
            return initial, "parallel_edge_fallback"
        t = np.linalg.solve(mat, q - p)[0]
        corners.append(p + t * v)
    quad = order_quad(corners)
    if cv2.contourArea(quad) < 0.7 * cv2.contourArea(initial) or cv2.contourArea(
        quad
    ) > 1.4 * cv2.contourArea(initial):
        return initial, "area_guard_fallback"
    # Small uniform outward allowance prevents source-color fringes at subpixel edges.
    return quad.astype(np.float32), "four_robust_boundary_lines"


def edge_refine(rgb, initial):
    q = initial.copy()
    logs = []
    x0 = max(0, int(q[:, 0].min()) - 30)
    x1 = min(rgb.shape[1], int(q[:, 0].max()) + 31)
    y0 = max(0, int(q[:, 1].min()) - 30)
    y1 = min(rgb.shape[0], int(q[:, 1].max()) + 31)
    region = rgb[y0:y1, x0:x1].astype(np.float32) / 255
    region = cv2.GaussianBlur(region, (0, 0), 0.6)
    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    short = min(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    for iteration in range(2):
        lines = []
        record = []
        center = q.mean(0)
        for i in range(4):
            p = q[i]
            end = q[(i + 1) % 4]
            d = (end - p) / np.linalg.norm(end - p)
            normal = np.array([-d[1], d[0]])
            if np.dot(center - p, normal) < 0:
                normal = -normal
            band = max(2.5, short * (0.10 if iteration == 0 else 0.035))
            offsets = np.linspace(-band, band, 61).astype(np.float32)
            t = np.linspace(0.12, 0.88, 100).astype(np.float32)
            base = p[None, :] + t[:, None] * (end - p)[None, :]
            loc = base[:, None, :] + offsets[None, :, None] * normal[None, None, :]
            strength = np.sqrt(np.mean((gx * normal[0] + gy * normal[1]) ** 2, axis=2))
            values = cv2.remap(
                strength,
                (loc[:, :, 0] - x0).astype(np.float32),
                (loc[:, :, 1] - y0).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            prior = np.exp(-0.5 * (offsets / (band * 0.55)) ** 2)
            weighted = values * prior[None, :]
            idx = weighted.argmax(1)
            raw = values[np.arange(len(t)), idx]
            pts = loc[np.arange(len(t)), idx]
            take = raw > max(0.005, float(np.percentile(raw, 35)))
            if take.sum() < 15:
                point = p
                direction = d
                status = "weak_edge_keep_sam"
            else:
                vx, vy, cx, cy = cv2.fitLine(
                    pts[take].astype(np.float32), cv2.DIST_HUBER, 0, 0.001, 0.001
                ).ravel()
                direction = np.array([vx, vy])
                point = np.array([cx, cy])
                angle = np.degrees(np.arccos(np.clip(abs(np.dot(direction, d)), 0, 1)))
                if angle > 9:
                    point = p
                    direction = d
                    status = "angle_guard_keep_sam"
                else:
                    status = "image_gradient_fit"
            lines.append((point, direction))
            record.append({"edge": i, "status": status, "samples": int(take.sum())})
        corners = []
        for i in range(4):
            p, v = lines[(i - 1) % 4]
            a, u = lines[i]
            m = np.column_stack((v, -u))
            if abs(np.linalg.det(m)) < 0.05:
                return initial, [{"status": "parallel_guard"}]
            corners.append(p + np.linalg.solve(m, a - p)[0] * v)
        candidate = order_quad(corners)
        if (
            not cv2.isContourConvex(candidate)
            or not 0.8 < cv2.contourArea(candidate) / cv2.contourArea(initial) < 1.2
        ):
            logs.append(
                {"iteration": iteration, "status": "geometry_guard_keep_previous"}
            )
            break
        q = candidate
        logs.append({"iteration": iteration, "edges": record})
    return q, logs


def assess_geometry(mask, quad):
    """Flag under-constrained cases; segmentation is not a full-plane oracle."""
    h, w = mask.shape
    polygon = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(polygon, np.rint(quad).astype(np.int32), 1)
    visible = mask > 0
    area = int(polygon.sum())
    coverage = float((visible & (polygon > 0)).sum() / max(area, 1))
    lengths = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    flags = []
    ys, xs = np.where(visible)
    if not len(xs):
        return {"flags": ["empty_mask"], "coverage": 0.0, "min_edge_px": 0.0}
    if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2:
        flags.append("partially_out_of_frame")
    if min(lengths) < 12:
        flags.append("surface_too_small")
    if coverage < 0.88:
        flags.append("occluded_or_non_planar_boundary")
    return {"flags": flags, "coverage": coverage, "min_edge_px": min(lengths)}


def template_geometry_flags(quad, artwork_aspect):
    """Shared still/video guard against stretching a visible sliver into a face.

    Strong genuine perspective can trigger this conservative gate. A manual
    complete-plane correction is needed; it is not a detector confidence score.
    """
    q = np.asarray(quad, np.float32)
    lengths = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    aspect = (lengths[0] + lengths[2]) / max(1.0, lengths[1] + lengths[3])
    if aspect / artwork_aspect < 0.55:
        return ["possible_occlusion_or_extreme_foreshortening"]
    return []
