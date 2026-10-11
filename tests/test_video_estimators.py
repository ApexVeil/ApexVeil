"""Product routes must share injectable estimators and per-native-ID geometry."""

import io
import json
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image

from apexveil.estimation import get_default_estimators
from apexveil.imaging.compositor import Template
from apexveil.imaging.geometry import SurfaceGeometry
from apexveil.imaging.photometry import TemporalPhotometry
from apexveil.video_pipeline import (
    LazyRGBFrames,
    add_declared_support,
    estimate_geometry_track,
    geometry_review_flags,
    render_surface_pair,
)


class CountingProvider:
    def __init__(self, provider, review=False):
        self.provider = provider
        self.calls = 0
        self.review = review

    def estimate(self, *args, **kwargs):
        self.calls += 1
        result = self.provider.estimate(*args, **kwargs)
        if self.review:
            result.diagnostics["needs_review"] = True
        return result


def scene():
    image = np.full((100, 220, 3), 70, np.uint8)
    image[30:70, 40:180] = np.random.default_rng(91).integers(
        40, 220, (40, 140, 3), dtype=np.uint8
    )
    mask = np.zeros((100, 220), np.uint8)
    mask[30:70, 40:180] = 1
    return image, mask


def provider_bundle():
    default = get_default_estimators()
    return replace(
        default,
        provider_id="test-injected",
        geometry=CountingProvider(default.geometry),
        boundary=CountingProvider(default.boundary, review=True),
        degradation=CountingProvider(default.degradation),
        appearance=CountingProvider(default.appearance),
    )


def test_geometry_provider_is_called_once_and_offline_retains_identity():
    image, mask = scene()
    bundle = provider_bundle()
    observations, estimates = estimate_geometry_track(
        [image] * 3, [mask] * 3, 3.5, estimators=bundle, identity=37
    )
    assert bundle.geometry.calls == 3
    assert all(e.quad is not None for e in estimates)
    assert all(e.diagnostics["association"]["identity"] == 37 for e in estimates)
    assert all(
        e.diagnostics["estimator"] == "offline_bidirectional_motion_consensus"
        for e in estimates
    )
    assert observations[0].visible_mask is mask


def test_product_renderer_passes_bundle_seed_support_and_boundary_review():
    image, mask = scene()
    bundle = provider_bundle()
    observations, estimates = estimate_geometry_track(
        [image], [mask], 3.5, estimators=bundle, identity=0
    )
    asset = np.full((40, 140, 3), 230, np.uint8)
    asset[12:28, 25:115] = 20
    template = Template(asset, np.ones((40, 140), np.float32), {})
    result = render_surface_pair(
        image,
        mask,
        observations[0],
        estimates[0],
        template,
        bundle,
        TemporalPhotometry(30),
        noise_seed=191,
    )
    baseline, fused, baseline_meta, meta, _, flags, sa, sb = result
    assert bundle.appearance.calls == 2
    assert bundle.boundary.calls == 2
    assert bundle.degradation.calls >= 1
    assert baseline_meta["noise_seed"] == meta["noise_seed"] == 191
    assert meta["temporal_appearance"]["geometry_filter"] == "offline_only"
    assert "boundary_needs_review" in flags
    for output, detail in ((baseline, sa), (fused, sb)):
        allowed = np.zeros(mask.shape, bool)
        add_declared_support(allowed, detail)
        np.testing.assert_array_equal(output[~allowed], image[~allowed])


def test_recovered_border_flag_is_historical_but_occlusion_remains_review():
    _, mask = scene()
    q = np.array([[40, 30], [179, 30], [179, 69], [40, 69]], np.float32)
    estimate = SurfaceGeometry(
        q,
        mask,
        0.6,
        [
            "partially_out_of_frame",
            "insufficient_boundary_support",
            "temporally_supported_full_plane",
            "offline_motion_refined",
            "occluded_or_non_planar_boundary",
        ],
    )
    assert geometry_review_flags(estimate) == ["occluded_or_non_planar_boundary"]


def test_lazy_png_frames_decode_the_requested_frame_and_release_cache(tmp_path):
    image, _ = scene()
    paths = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full_like(image, i * 50)).save(path)
        paths.append(path)
    frames = LazyRGBFrames(paths)
    assert frames[2].mean() == 100
    assert frames[0].mean() == 0
    frames.close()
    assert frames.cached is None


def test_montage_keeps_alias_geometry_separate_and_uses_injected_provider(
    tmp_path, monkeypatch
):
    import apexveil.montage_pipeline as montage

    image = np.full((720, 1280, 3), 75, np.uint8)
    texture = np.random.default_rng(193).integers(70, 220, (55, 160, 3), dtype=np.uint8)
    image[330:385, 550:710] = texture
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[330:385, 550:710] = 255
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    class Capture:
        def __init__(self, *_):
            self.index = 0

        def set(self, _, index):
            self.index = int(index)

        def read(self):
            if self.index >= 4:
                return False, None
            self.index += 1
            return True, bgr.copy()

        def release(self):
            pass

    class Writer:
        def __init__(self):
            self.stdin = io.BytesIO()

        def kill(self):
            pass

        def wait(self):
            return 0

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(montage, "raw_writer", lambda *args: Writer())
    monkeypatch.setattr(montage, "finish_writer", lambda writer: None)
    monkeypatch.setattr(
        montage, "verify_encoded_output", lambda *args: {"frame_count": 4}
    )
    tracking = tmp_path / "tracking" / "shot_000"
    records = []
    for i in range(4):
        identity = i // 2
        frame = tracking / f"frame_{i:06d}"
        frame.mkdir(parents=True)
        name = f"object_{identity:04d}.png"
        Image.fromarray(mask).save(frame / name)
        records.append(
            {"frame_index": i, "objects": [{"object_id": identity, "mask": name}]}
        )
    (tracking / "report.json").write_text(
        json.dumps({"frames": records, "native_error": None})
    )
    rgba = np.full((40, 140, 4), 230, np.uint8)
    rgba[:, :, 3] = 255
    template = tmp_path / "template.png"
    Image.fromarray(rgba).save(template)
    shot = {
        "id": "shot_000",
        "frame_count": 4,
        "start_frame": 0,
        "target_id": 0,
        "primary_ids": [0, 1],
        "exclude_ids": [],
        "kind": "shot",
    }
    bundle = provider_bundle()
    report = montage.render_shot(str(tmp_path), shot, str(template), estimators=bundle)
    rows = [
        json.loads(line)
        for line in (tmp_path / "render" / "shot_000.jsonl").read_text().splitlines()
    ]
    assert bundle.geometry.calls == 8
    assert report["signature"]["estimator"]["provider_id"] == "test-injected"
    assert [r["target_id"] for r in rows] == [0, 0, 1, 1]
    assert rows[2]["primary_alias_switched"]
    for row in rows:
        assert row["geometry_state"]["association"]["identity"] == row["target_id"]
        assert row["outside_edit_support_changed_pixels"] == 0
        assert row["status"] == "needs_review"
        assert row["apexveil_replaced"]
    seeds = [r["apexveil_appearance"]["noise_seed"] for r in rows]
    assert len(set(seeds)) == 4


def test_injected_geometry_needs_no_private_diagnostic_fields_for_motion():
    image, mask = scene()
    q = np.array([[40, 30], [179, 30], [179, 69], [40, 69]], np.float32)

    class Geometry:
        def estimate(self, rgb, mask, aspect):
            return SurfaceGeometry(q.copy(), mask, 0.9, [], {"provider": "test"})

    bundle = replace(
        get_default_estimators(), geometry=Geometry(), provider_id="custom"
    )
    _, estimates = estimate_geometry_track(
        [image] * 3, [mask] * 3, 3.5, estimators=bundle, identity=4
    )
    assert len(estimates[1].diagnostics["anchors"]) == 3
    assert estimates[1].diagnostics["provider"] == "test"


def test_injected_plane_cannot_override_empty_native_visibility():
    image, mask = scene()
    q = np.array([[40, 30], [179, 30], [179, 69], [40, 69]], np.float32)

    class Geometry:
        def estimate(self, rgb, mask, aspect):
            return SurfaceGeometry(q.copy(), mask, 0.9)

    bundle = replace(
        get_default_estimators(), geometry=Geometry(), provider_id="custom"
    )
    _, estimates = estimate_geometry_track(
        [image], [np.zeros_like(mask)], 3.5, estimators=bundle
    )
    assert estimates[0].quad is None
