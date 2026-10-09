"""Motion-compensated surface state for bounded continuous-shot experiments.

Geometry fuses same-time LK/RANSAC predictions with current observations. This
avoids the lag introduced by averaging corners in screen coordinates. All
visibility remains the current frame's mask; flow never fills occluders.
"""

from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import order_quad


@dataclass
class TemporalEstimate:
    quad: np.ndarray | None
    diagnostics: dict


def valid_quad(quad, reference=None):
    if quad is None:
        return False
    q = np.asarray(quad, np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all() or not cv2.isContourConvex(q):
        return False
    area = cv2.contourArea(q)
    if area < 16:
        return False
    return reference is None or 0.55 < area / max(1.0, cv2.contourArea(reference)) < 1.8


def flow_prediction(previous, current, mask, quad, current_mask=None):
    """Track textured visible surface points with forward/backward rejection."""
    points = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=100,
        qualityLevel=0.005,
        minDistance=2,
        mask=(mask > 0).astype(np.uint8) * 255,
        blockSize=3,
    )
    details = {
        "method": "unavailable",
        "feature_count": 0,
        "inliers": 0,
        "confidence": 0.0,
    }
    if points is None or len(points) < 5:
        return None, details
    details["feature_count"] = len(points)
    moved, good, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None:
        return None, details
    returned, backward_good, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        moved,
        None,
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if returned is None:
        return None, details
    fb = np.linalg.norm(returned[:, 0] - points[:, 0], axis=1)
    take = good[:, 0].astype(bool) & backward_good[:, 0].astype(bool) & (fb < 1.25)
    height, width = current.shape
    destinations = moved[:, 0]
    inside = (
        (destinations[:, 0] >= 0)
        & (destinations[:, 1] >= 0)
        & (destinations[:, 0] < width)
        & (destinations[:, 1] < height)
    )
    if current_mask is not None:
        ix = np.clip(np.rint(destinations[:, 0]).astype(int), 0, width - 1)
        iy = np.clip(np.rint(destinations[:, 1]).astype(int), 0, height - 1)
        current_visible = inside & (current_mask[iy, ix] > 0)
    else:
        current_visible = inside
    details["current_visibility_rejected"] = int((take & ~current_visible).sum())
    take &= current_visible
    if take.sum() < 5:
        return None, details
    a, b = points[take, 0], moved[take, 0]
    h = None
    if len(a) >= 8:
        h, inliers = cv2.findHomography(a, b, cv2.RANSAC, 1.5)
        if h is not None and int(inliers.sum()) >= 8:
            details["method"] = "lk_ransac_homography"
    if h is None or details["method"] == "unavailable":
        affine, inliers = cv2.estimateAffinePartial2D(
            a, b, method=cv2.RANSAC, ransacReprojThreshold=1.5
        )
        if affine is None or int(inliers.sum()) < 5:
            return None, details
        h = np.vstack([affine, [0.0, 0.0, 1.0]])
        details["method"] = "lk_ransac_similarity_fallback"
    prediction = cv2.perspectiveTransform(quad[None].astype(np.float32), h)[0]
    if not valid_quad(prediction, quad):
        details["method"] = "invalid_motion_rejected"
        return None, details
    inlier_count = int(inliers.sum())
    median_fb = float(np.median(fb[take]))
    confidence = min(1.0, inlier_count / 15) * (inlier_count / len(a)) / (1 + median_fb)
    details.update(
        inliers=inlier_count,
        median_forward_backward_error_px=median_fb,
        confidence=float(confidence),
        homography=h.tolist(),
    )
    return prediction, details


class TemporalGeometry:
    def __init__(self, maximum_prediction_frames=5):
        self.previous_gray = None
        self.previous_mask = None
        self.previous_quad = None
        self.prediction_age = 0
        self.maximum_prediction_frames = maximum_prediction_frames
        self.awaiting_observation_recovery = False

    @staticmethod
    def _mask_agreement(quad, visible_mask):
        """Current-frame evidence, independent of the flow fit's inlier score."""
        corners = np.rint(quad).astype(np.int32)
        h, w = visible_mask.shape
        x0, y0 = np.maximum(corners.min(axis=0), 0)
        x1, y1 = np.minimum(corners.max(axis=0) + 1, [w, h])
        if x1 <= x0 or y1 <= y0:
            return 0.0
        polygon = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(polygon, corners - [x0, y0], 1)
        local = visible_mask[y0:y1, x0:x1] > 0
        intersection = int(((polygon > 0) & local).sum())
        union = int(polygon.sum()) + int(np.count_nonzero(visible_mask)) - intersection
        return intersection / max(1, union)

    def _bounded_observation_update(
        self, prediction, observation, visible_mask, flow_confidence, short, details
    ):
        """Correct the already-motion-compensated plane, never screen-space EMA."""
        prediction_support = self._mask_agreement(prediction, visible_mask)
        observation_support = self._mask_agreement(observation, visible_mask)
        advantage = observation_support - prediction_support
        details.update(
            prediction_mask_agreement=prediction_support,
            observation_mask_agreement=observation_support,
        )
        # A high-inlier LK estimate can still extrapolate poorly at the corners.
        # Conversely a fitted corner can jump off the visible surface. Use both
        # independent current-mask agreement and tracker confidence to decide.
        if advantage < -0.05:
            details["flags"].append("observation_rejected_by_visibility")
            details["observation_weight"] = 0.0
            details["correction_decision"] = "reject_observation"
            self.prediction_age += 1
            if self.prediction_age > self.maximum_prediction_frames:
                details["flags"].append("prediction_budget_exhausted")
                self.awaiting_observation_recovery = True
                return None
            details["flags"].append("flow_only_geometry_prediction")
            return prediction
        confidence = float(np.clip(flow_confidence, 0.0, 1.0))
        weight = float(
            np.clip(
                0.30 + 0.40 * (1 - confidence) + 1.5 * max(0.0, advantage), 0.20, 0.85
            )
        )
        # The cap expands only if motion evidence weakens or current visible
        # pixels strongly favor the observation. Actual vehicle motion is already
        # represented in prediction and is not capped by this innovation limit.
        limit = max(2.0, float(short) * (0.08 + 0.15 * (1 - confidence)))
        limit += float(short) * min(0.40, max(0.0, advantage))
        correction = (observation - prediction) * weight
        norms = np.linalg.norm(correction, axis=1)
        correction *= np.minimum(1.0, limit / np.maximum(norms, 1e-6))[:, None]
        candidate = prediction + correction
        details.update(
            observation_weight=weight,
            innovation_limit_px=float(limit),
            applied_innovation_max_px=float(np.linalg.norm(correction, axis=1).max()),
            correction_decision="bounded_observation_update",
        )
        if np.any(norms > limit):
            details["flags"].append("observation_correction_limited")
        if not valid_quad(candidate, prediction):
            details["flags"].append("invalid_corrected_quad_rejected")
            self.prediction_age += 1
            if self.prediction_age > self.maximum_prediction_frames:
                details["flags"].append("prediction_budget_exhausted")
                self.awaiting_observation_recovery = True
                return None
            details["flags"].append("flow_only_geometry_prediction")
            return prediction
        self.prediction_age = 0
        self.awaiting_observation_recovery = False
        return candidate

    def step(self, rgb, visible_mask, observed_quad, observation_reliable=True):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        observation = (
            np.asarray(observed_quad, np.float32) if observed_quad is not None else None
        )
        if observation is not None and not valid_quad(observation):
            observation = None
        details = {
            "status": "needs_review",
            "flags": [],
            "flow": {},
            "observation_weight": None,
        }
        prediction = None
        cut = False
        if self.previous_gray is not None:
            difference = float(
                np.mean(np.abs(gray.astype(np.float32) - self.previous_gray)) / 255
            )
            cut = difference > 0.30
            details["global_frame_difference"] = difference
            if cut:
                details["flags"].append("possible_cut_or_flash_state_reset")
            elif self.previous_quad is not None and self.previous_mask.any():
                prediction, details["flow"] = flow_prediction(
                    self.previous_gray,
                    gray,
                    self.previous_mask,
                    self.previous_quad,
                    current_mask=visible_mask,
                )
        if cut:
            self.prediction_age = 0
        if prediction is not None and details["flow"].get("confidence", 0) >= 0.3:
            if observation is not None and observation_reliable:
                corner_innovations = np.linalg.norm(observation - prediction, axis=1)
                residual = float(np.median(corner_innovations))
                short = min(
                    np.linalg.norm(prediction[(i + 1) % 4] - prediction[i])
                    for i in range(4)
                )
                details["observation_motion_residual_px"] = residual
                details["maximum_corner_innovation_px"] = float(
                    corner_innovations.max()
                )
                corner_outlier = float(corner_innovations.max()) > max(
                    3.0, float(short) * 0.10
                ) and float(corner_innovations.max()) > 3 * (residual + 0.25)
                if corner_outlier:
                    details["flags"].append("single_corner_innovation_outlier")
                if residual < max(4.0, float(short) * 0.18) and not corner_outlier:
                    weight = float(
                        np.clip(0.45 - 0.25 * details["flow"]["confidence"], 0.20, 0.45)
                    )
                    q = prediction * (1 - weight) + observation * weight
                    details["observation_weight"] = weight
                    details["status"] = "motion_fused"
                    self.prediction_age = 0
                else:
                    details["flags"].append("flow_observation_disagreement")
                    details["status"] = "needs_review"
                    q = self._bounded_observation_update(
                        prediction,
                        observation,
                        visible_mask,
                        details["flow"]["confidence"],
                        short,
                        details,
                    )
            elif (
                self.prediction_age < self.maximum_prediction_frames
                and visible_mask.any()
            ):
                q = prediction
                self.prediction_age += 1
                details["status"] = "needs_review"
                details["flags"].append("flow_only_geometry_prediction")
            else:
                q = None
                details["flags"].append("insufficient_geometry")
        elif observation is not None and observation_reliable:
            if (
                self.awaiting_observation_recovery
                and self._mask_agreement(observation, visible_mask) < 0.85
            ):
                q = None
                details["flags"].append("observation_recovery_needs_review")
            else:
                q = observation
                details["status"] = (
                    "initialized"
                    if self.previous_gray is None or cut
                    else "observation_only"
                )
                if cut:
                    details["status"] = "needs_review"
                self.prediction_age = 0
                self.awaiting_observation_recovery = False
        else:
            q = None
            details["flags"].append("insufficient_geometry")
        if q is not None:
            q = order_quad(q)
        details["prediction_age"] = self.prediction_age
        self.previous_gray = gray
        self.previous_mask = (visible_mask > 0).astype(np.uint8)
        self.previous_quad = q
        if q is not None and prediction is not None:
            details["fused_motion_residual_px"] = float(
                np.median(np.linalg.norm(q - prediction, axis=1))
            )
        return TemporalEstimate(q, details)


def rectified_visible_reference(rgb, mask, quad, size):
    """Remove invisible foreground before it can enter the illumination state."""
    width, height = size
    canonical = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32
    )
    inverse = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), canonical)
    rect = cv2.warpPerspective(rgb, inverse, size, borderMode=cv2.BORDER_REFLECT)
    visible = cv2.warpPerspective(
        (mask > 0).astype(np.uint8), inverse, size, flags=cv2.INTER_NEAREST
    )
    coverage = float(visible.mean())
    if coverage < 0.5:
        return None, coverage
    missing = (visible == 0).astype(np.uint8)
    if missing.any():
        rect = cv2.inpaint(rect, missing * 255, 3, cv2.INPAINT_TELEA)
    return rect, coverage


class TemporalAppearance:
    def __init__(self):
        self.reference = None
        self.softness = None

    def reset(self):
        self.reference = None
        self.softness = None

    def step(self, reference, sigma):
        diagnostics = {"appearance_weight_current": 1.0, "flags": []}
        if reference is None:
            diagnostics["flags"].append("insufficient_visible_appearance")
            return None, sigma, diagnostics
        current = reference.astype(np.float32)
        if self.reference is not None:
            scale = max(1.0, reference.shape[0] * 0.04)
            old = cv2.GaussianBlur(self.reference, (0, 0), scale)
            new = cv2.GaussianBlur(current, (0, 0), scale)
            relative_change = float(
                np.percentile(np.abs(new - old), 90) / max(12.0, np.percentile(old, 90))
            )
            # Respond quickly to real moving shadows/exposure instead of
            # freezing them for the sake of a lower temporal-variance metric.
            weight = 0.85 if relative_change > 0.15 else 0.35
            diagnostics.update(
                appearance_weight_current=weight, low_frequency_change=relative_change
            )
            self.reference = current * weight + self.reference * (1 - weight)
        else:
            self.reference = current
        if self.softness is None or abs(sigma - self.softness) > 3:
            self.softness = float(sigma)
        else:
            self.softness = 0.4 * float(sigma) + 0.6 * self.softness
        return (
            np.clip(np.rint(self.reference), 0, 255).astype(np.uint8),
            self.softness,
            diagnostics,
        )
