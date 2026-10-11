"""Robust planar geometry, separate from visible segmentation."""

from dataclasses import dataclass, field
from itertools import combinations

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
    short_edge = min(
        np.linalg.norm(initial[(i + 1) % 4] - initial[i]) for i in range(4)
    )
    allowance = min(1.0, float(short_edge) * 0.02)
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
        lines.append((point - normal * allowance, direction))
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


def edge_refine(rgb, initial, visible_mask=None):
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
            if short < 48:
                band = max(0.75, short * (0.035 if iteration == 0 else 0.015))
            else:
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
        if visible_mask is not None:
            prior_support = visible_coverage(visible_mask, q)
            new_support = visible_coverage(visible_mask, candidate)
            if new_support < prior_support - 0.02:
                logs.append(
                    {
                        "iteration": iteration,
                        "status": "visibility_support_guard_keep_previous",
                        "prior_coverage": prior_support,
                        "candidate_coverage": new_support,
                    }
                )
                break
        q = candidate
        logs.append({"iteration": iteration, "edges": record})
    return q, logs


def visible_coverage(mask, quad):
    """Measure visible support locally without allocating a full-frame polygon."""
    h, w = mask.shape
    vertices = np.rint(quad).astype(np.int32)
    x0, y0 = np.maximum(vertices.min(axis=0), 0)
    x1, y1 = np.minimum(vertices.max(axis=0) + 1, [w, h])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    polygon = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(polygon, vertices - [x0, y0], 1)
    return float(
        ((mask[y0:y1, x0:x1] > 0) & (polygon > 0)).sum() / max(int(polygon.sum()), 1)
    )


def assess_geometry(mask, quad):
    """Flag under-constrained cases; segmentation is not a full-plane oracle."""
    h, w = mask.shape
    visible = mask > 0
    coverage = visible_coverage(mask, quad)
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
    """Validate a projective plane, never infer occlusion from apparent aspect.

    Any nondegenerate convex quadrilateral can project the artwork rectangle.
    Visibility evidence belongs to the estimator, not an apparent aspect gate.
    """
    if not np.isfinite(artwork_aspect) or artwork_aspect <= 0:
        raise ValueError("Artwork aspect must be finite and positive")
    try:
        q = order_quad(quad)
    except (ValueError, cv2.error):
        return ["degenerate_projected_surface"]
    edges = np.roll(q, -1, axis=0) - q
    lengths = np.linalg.norm(edges, axis=1)
    cross = np.abs(np.cross(edges, np.roll(edges, -1, axis=0)))
    sine = cross / np.maximum(lengths * np.roll(lengths, -1), 1e-6)
    if min(lengths) < 2 or min(sine) < 0.025:
        return ["ill_conditioned_projected_surface"]
    return []


@dataclass
class SurfaceGeometry:
    """Full-plane hypothesis and evidence; never a replacement mask.

    Confidence is an engineering evidence score, not a calibrated probability.
    visible_mask is always the unmodified current native segmentation.
    """

    quad: np.ndarray | None
    visible_mask: np.ndarray
    confidence: float
    flags: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _boundary_support(mask, quad):
    # Bound the distance transform to the surface ROI, including exterior pixels.
    h0, w0 = mask.shape
    low = np.maximum(np.floor(quad.min(0)).astype(int) - 6, 0)
    high = np.minimum(np.ceil(quad.max(0)).astype(int) + 7, [w0, h0])
    if np.any(high <= low):
        return [0.0] * 4
    mask = mask[low[1] : high[1], low[0] : high[0]]
    quad = quad - low
    binary = (mask > 0).astype(np.uint8)
    boundary = binary - cv2.erode(binary, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    short = min(np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4))
    tolerance = max(1.5, min(3.0, short * 0.025))
    supports = []
    h, w = mask.shape
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        points = a + np.linspace(0.08, 0.92, 80)[:, None] * (b - a)
        x, y = np.rint(points).astype(int).T
        valid = (x >= 1) & (x < w - 1) & (y >= 1) & (y < h - 1)
        close = np.zeros(len(points), bool)
        close[valid] = distance[y[valid], x[valid]] <= tolerance
        supports.append(float(close.mean()))
    return supports


def visible_containment(mask, quad):
    """Fraction of observed pixels explained by a plane (occluders are absent)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(local, np.rint(quad - [x0, y0]).astype(np.int32), 1)
    return float(local[ys - y0, xs - x0].mean())


def _supported_enclosing_quad(mask):
    """Intersect observed hull sides, allowing a modest missing corner."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(contours))
    poly = cv2.approxPolyDP(hull, 0.008 * cv2.arcLength(hull, True), True)[:, 0]
    if not 4 <= len(poly) <= 10:
        return None
    area = cv2.contourArea(hull)
    best = None
    for ids in combinations(range(len(poly)), 4):
        lines = [
            (poly[i].astype(float), (poly[(i + 1) % len(poly)] - poly[i]).astype(float))
            for i in ids
        ]
        corners = []
        for j in range(4):
            a, v = lines[j - 1]
            b, u = lines[j]
            mat = np.column_stack((v, -u))
            if abs(np.linalg.det(mat)) < 1e-4:
                break
            corners.append(a + np.linalg.solve(mat, b - a)[0] * v)
        if len(corners) != 4:
            continue
        try:
            q = order_quad(corners)
        except ValueError:
            continue
        ratio = cv2.contourArea(q) / max(area, 1)
        if not 0.97 <= ratio <= 1.6:
            continue
        if min(cv2.pointPolygonTest(q, tuple(map(float, p)), True) for p in poly) < -2:
            continue
        support = _boundary_support(mask, q)
        if min(support) < 0.30:
            continue
        score = np.mean(support) - 0.25 * (ratio - 1)
        if best is None or score > best[0]:
            best = (score, q)
    return None if best is None else best[1]


def estimate_surface_geometry(
    rgb, visible_mask, template_aspect_ratio, *, observed_quad=None
):
    """Estimate full-plane geometry without assuming a frontal view.

    Four supported boundary lines identify a plane despite internal foreground
    holes or a modest missing corner. Frame clipping is under-constrained in a
    single image and returns no plane; offline motion can recover it from a
    supported neighbor. A rectangular visible sliver remains fundamentally
    ambiguous without additional image or temporal evidence.
    """
    mask = np.asarray(visible_mask)
    if rgb is not None and np.asarray(rgb).shape[:2] != mask.shape:
        raise ValueError("RGB frame and visible mask dimensions must match")
    if mask.ndim != 2:
        raise ValueError("Visible mask must have shape (height, width)")
    if not np.isfinite(template_aspect_ratio) or template_aspect_ratio <= 0:
        raise ValueError("Template aspect must be finite and positive")
    binary = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary) < 16:
        return SurfaceGeometry(None, mask, 0.0, ["insufficient_visible_surface"])
    if observed_quad is not None:
        try:
            quad = order_quad(observed_quad)
        except ValueError:
            return SurfaceGeometry(None, mask, 0.0, ["invalid_observation"])
        method = "provided_observation"
    else:
        try:
            quad = _supported_enclosing_quad(binary)
            method = "supported_boundary_intersections"
            if quad is None:
                quad, method = fit_quad(binary)
        except (ValueError, cv2.error):
            return SurfaceGeometry(None, mask, 0.0, ["degenerate_visible_surface"])
    edge_logs = []
    if (
        rgb is not None
        and "partially_out_of_frame" not in assess_geometry(binary, quad)["flags"]
    ):
        try:
            refined, edge_logs = edge_refine(rgb, quad, visible_mask=binary)
        except (ValueError, cv2.error):
            refined = quad
            edge_logs = [{"status": "invalid_rgb_refinement_keep_boundary"}]
        # RGB edges contribute only when independently supported by segmentation.
        if min(_boundary_support(binary, refined)) >= min(
            _boundary_support(binary, quad)
        ):
            quad = refined
    assessment = assess_geometry(binary, quad)
    flags = list(assessment["flags"])
    flags.extend(template_geometry_flags(quad, template_aspect_ratio))
    support = _boundary_support(binary, quad)
    containment = visible_containment(binary, quad)
    confidence = float(
        np.clip(np.mean(support) * min(support) ** 0.5 * containment, 0, 1)
    )
    diagnostics = {
        **assessment,
        "method": method,
        "boundary_support": support,
        "visible_containment": containment,
        "candidate_quad": quad.tolist(),
        "confidence_semantics": "uncalibrated_boundary_evidence",
        "rgb_edge_refinement": edge_logs,
        "protected_mask": "unavailable",
        "full_plane_status": "boundary_hypothesis_not_semantic_confirmation",
        "completeness_confidence": None,
    }
    if min(support) < 0.35 or containment < 0.92:
        flags.append("insufficient_boundary_support")
    blocking = {
        "partially_out_of_frame",
        "insufficient_boundary_support",
        "ill_conditioned_projected_surface",
        "degenerate_projected_surface",
    }
    if blocking.intersection(flags):
        return SurfaceGeometry(None, mask, min(confidence, 0.25), flags, diagnostics)
    if assessment["min_edge_px"] < 12:
        confidence *= assessment["min_edge_px"] / 12
    return SurfaceGeometry(quad, mask, confidence, flags, diagnostics)
