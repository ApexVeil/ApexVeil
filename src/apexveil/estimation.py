"""Replaceable estimators feeding a deterministic, artwork-preserving renderer.

The installed provider performs input-derived estimation and optimization only.
A future learned provider may predict these same geometric/material quantities;
no neural runtime, weights, pixel generator, or paid service is loaded here.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .imaging.appearance import AppearanceEstimate
    from .imaging.boundary import BoundaryEstimate
    from .imaging.degradation import DegradationEstimate
    from .imaging.geometry import SurfaceGeometry


@runtime_checkable
class GeometryEstimator(Protocol):
    def estimate(
        self,
        rgb: np.ndarray,
        visible_mask: np.ndarray,
        template_aspect_ratio: float,
        *,
        observed_quad=None,
    ) -> "SurfaceGeometry": ...


@runtime_checkable
class BoundaryEstimator(Protocol):
    def estimate(
        self,
        rgb: np.ndarray,
        visible_mask: np.ndarray,
        quad: np.ndarray,
        *,
        protected_mask=None,
        blur_sigma: float = 0.35,
    ) -> "BoundaryEstimate": ...


@runtime_checkable
class DegradationEstimator(Protocol):
    def estimate(
        self, rgb: np.ndarray, mask: np.ndarray, quad: np.ndarray
    ) -> "DegradationEstimate": ...


@runtime_checkable
class AppearanceEstimator(Protocol):
    def estimate(
        self,
        asset: np.ndarray,
        original_rect: np.ndarray,
        reference_percentile=99.5,
        match_black=True,
        valid_mask=None,
        context_tint=None,
        ink_resolved=True,
        context_black_level=None,
        parameter_filter=None,
    ) -> "AppearanceEstimate": ...


class DeterministicGeometry:
    def estimate(self, rgb, visible_mask, template_aspect_ratio, *, observed_quad=None):
        from .imaging.geometry import estimate_surface_geometry

        return estimate_surface_geometry(
            rgb, visible_mask, template_aspect_ratio, observed_quad=observed_quad
        )


class DeterministicBoundary:
    def estimate(
        self, rgb, visible_mask, quad, *, protected_mask=None, blur_sigma=0.35
    ):
        from .imaging.boundary import estimate_boundary

        return estimate_boundary(
            rgb,
            visible_mask,
            quad,
            protected_mask=protected_mask,
            blur_sigma=blur_sigma,
        )


class DeterministicDegradation:
    def estimate(self, rgb, mask, quad):
        from .imaging.degradation import estimate_degradation

        return estimate_degradation(rgb, mask, quad)


class DeterministicAppearance:
    def estimate(self, asset, original_rect, **kwargs):
        from .imaging.appearance import estimate_appearance

        return estimate_appearance(asset, original_rect, **kwargs)


@dataclass(frozen=True)
class Estimators:
    geometry: GeometryEstimator
    boundary: BoundaryEstimator
    degradation: DegradationEstimator
    appearance: AppearanceEstimator
    provider_id: str = "deterministic-v1"
    learned: bool = False

    def __post_init__(self):
        if not self.provider_id:
            raise ValueError("Estimator provider_id is required for provenance")
        for name in ("geometry", "boundary", "degradation", "appearance"):
            if not callable(getattr(getattr(self, name), "estimate", None)):
                raise TypeError(f"{name} must implement estimate()")

    def metadata(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "learned": self.learned,
            "generates_artwork_pixels": False,
        }


_DEFAULT = Estimators(
    DeterministicGeometry(),
    DeterministicBoundary(),
    DeterministicDegradation(),
    DeterministicAppearance(),
)


def get_default_estimators() -> Estimators:
    """Stateless provider adapters are safe to reuse; temporal state is per shot."""
    return _DEFAULT
