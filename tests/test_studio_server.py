import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from apexveil.web.server import Studio, StudioConfig, StudioHTTPServer, contained_file


@pytest.fixture
def running_studio(tmp_path):
    config = tmp_path / "default.toml"
    config.write_text('[project]\nroot="."\n[model]\ncheckpoint="unused.pt"\n')
    studio = Studio(StudioConfig(config, tmp_path / "out", advisor_profile="off"))
    server = StudioHTTPServer(("127.0.0.1", 0), studio)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield studio, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    studio.close()
    thread.join(timeout=2)


def test_upload_is_scoped_and_unknown_or_escaping_files_are_unavailable(running_studio):
    studio, base = running_studio
    with urlopen(
        Request(
            base + "/api/assets?name=../../source.png",
            data=b"private-bytes",
            method="POST",
        )
    ) as response:
        record = json.load(response)
    assert record["name"] == "source.png"
    assert record["kind"] == "image"
    assert studio.assets[record["asset_id"]]["_path"].parent == studio.root / "assets"
    assert all(not k.startswith("_") for k in record)
    for path in ["/files/../../config.toml", "/api/jobs/" + "f" * 32]:
        with pytest.raises(HTTPError) as error:
            urlopen(base + path)
        assert error.value.code == 404


def test_other_websites_cannot_trigger_gpu_work(running_studio):
    _, base = running_studio
    with pytest.raises(HTTPError) as error:
        urlopen(
            Request(
                base + "/api/assets?name=image.png",
                data=b"x",
                headers={"Origin": "https://unrelated.example"},
            )
        )
    assert error.value.code == 403
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/api/config", headers={"Host": "unrelated.example"}))
    assert error.value.code == 403


def test_video_downloads_support_precise_byte_ranges(running_studio):
    studio, base = running_studio
    clip = studio.root / "clip.mp4"
    clip.write_bytes(bytes(range(256)) * 8)
    with urlopen(
        Request(base + "/files/clip.mp4", headers={"Range": "bytes=100-199"})
    ) as response:
        assert response.status == 206
        assert response.read() == clip.read_bytes()[100:200]
        assert response.headers["Content-Range"] == "bytes 100-199/2048"
    with pytest.raises(HTTPError) as error:
        urlopen(Request(base + "/files/clip.mp4", headers={"Range": "bytes=9999-"}))
    assert error.value.code == 416


def test_assets_cannot_escape_via_symlink(tmp_path):
    root = tmp_path / "private"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("not public")
    (root / "escape").symlink_to(secret)
    with pytest.raises(FileNotFoundError):
        contained_file(root, "escape")


def test_config_hides_operator_model_and_network_settings(running_studio):
    _, base = running_studio
    with urlopen(base + "/api/config") as response:
        config = json.load(response)
    assert config["max_video_seconds"] == 8
    assert config["ai_available"] is False
    assert "proxy_url" not in config and "advisor_profile" not in config


def test_replacing_selection_job_releases_memory_before_admission(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    media = studio.root / "assets/image.png"
    media.write_bytes(b"placeholder")
    studio.register_asset(media, "photo.png", "image", "a" * 32)
    previous = None
    for _ in range(6):
        data = {"media_id": "a" * 32, "mode": "blur", "supersedes_id": previous}
        new = studio.create_job(data)["id"]
        if previous:
            assert studio.jobs[previous]["state"] == "cancelled"
            assert "_prepared" not in studio.jobs[previous]
        studio.jobs[new]["state"] = "awaiting_selection"
        studio.jobs[new]["_prepared"] = {"large_arrays": "retained until cancellation"}
        previous = new


def test_model_catalog_and_default_are_explicit(running_studio):
    _, base = running_studio
    config = json.load(urlopen(base + "/api/config"))
    assert config["ai_defaults"] == {"model": "gpt-6.1-sol", "effort": "medium"}
    assert [m["id"] for m in config["ai_models"]] == [
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6-astra",
    ]
    assert all(
        m["efforts"] == ["low", "medium", "high", "xhigh", "max"]
        for m in config["ai_models"]
    )


def test_render_time_advisor_changes_keep_detection_and_reach_workflow(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    studio.config.advisor_profile = "sol-medium"
    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    media = studio.root / "assets/source.png"
    media.write_bytes(b"input")
    studio.config.default_template = media
    studio.register_asset(media, "source.png", "image", "a" * 32)
    key = studio.create_job({"media_id": "a" * 32, "ai_assist": True})["id"]
    job = studio.jobs[key]
    prepared = object()
    job.update(
        state="awaiting_selection",
        public={"targets": [{"id": 0}]},
        _prepared=prepared,
        _model_load_seconds=0,
        _prepare_wall=0,
        _queue_seconds=0,
    )
    studio.request_render(
        key,
        {
            "target_id": 0,
            "ai_model": "gpt-6-astra",
            "ai_effort": "xhigh",
            "ai_assist": True,
        },
    )
    assert job["_prepared"] is prepared
    assert job["ai_effort"] == "xhigh"

    def fake_render(p, template, output, **kwargs):
        assert p is prepared
        assert kwargs["advisor_profile"] == "astra-xhigh"
        output.mkdir()
        report = output / "report.json"
        report.write_text("{}")
        return {
            "output_path": output / "result.png",
            "report_path": report,
            "timings": {},
            "warnings": [],
            "media_type": "image",
        }

    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_render)
    studio.render(job, 0)
    assert job["state"] == "completed"
    assert job["result"]["ai_selection"] == {
        "enabled": True,
        "model": "gpt-6-astra",
        "effort": "xhigh",
    }


def test_estimated_duration_is_not_completion(running_studio, monkeypatch):
    studio, _ = running_studio
    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    media = studio.root / "assets/source.png"
    media.write_bytes(b"x")
    studio.register_asset(media, "source.png", "image", "a" * 32)
    key = studio.create_job({"media_id": "a" * 32, "mode": "blur"})["id"]
    job = studio.jobs[key]
    job.update(
        state="rendering",
        phase="render",
        _phase_started=0.0001,
        estimated_phase_seconds=1,
    )
    snapshot = studio.snapshot(key)
    assert snapshot["phase_elapsed_seconds"] > 1
    assert snapshot["state"] == "rendering"
    assert snapshot["result"] is None


def test_waiting_image_selection_survives_service_restart(running_studio, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    studio, _ = running_studio
    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    media = studio.root / "assets" / ("b" * 32 + ".png")
    Image.new("RGB", (8, 6), (30, 40, 50)).save(media)
    studio.register_asset(media, "photo.png", "image", "b" * 32)
    key = studio.create_job({"media_id": "b" * 32, "mode": "blur"})["id"]
    job = studio.jobs[key]
    folder = job["_directory"] / "prepare"
    (folder / "native-masks").mkdir(parents=True)
    Image.fromarray(np.ones((6, 8), np.uint8) * 255).save(
        folder / "native-masks/plate-0.png"
    )
    (folder / "prepare-report.json").write_text(
        json.dumps({"source_sha256": hashlib.sha256(media.read_bytes()).hexdigest()})
    )
    studio.update(job, state="awaiting_selection", public={"targets": [{"id": 0}]})
    restored = Studio(studio.config)
    try:
        pending = restored.jobs[key]
        assert pending["state"] == "awaiting_selection"
        assert pending["_resume_selection"]
        restored.segmenter = SimpleNamespace(metadata={})
        restored.restore_image_selection(pending)
        assert pending["_prepared"]["timings"]["segmentation_cache_reused"]
        assert pending["_prepared"]["private"]["masks"][0].sum() == 48
    finally:
        restored.close()


def prepared_image_job(studio, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image

    calls = []

    class FakeSegmenter:
        def __init__(self):
            self.metadata = {"fixture": True}

        def detect(self, image, *, prompt, refine=True):
            calls.append(prompt)
            mask = np.zeros((image.height, image.width), np.uint8)
            mask[8:16, 8:30] = 1
            return [SimpleNamespace(object_id=0, visible_mask=mask, score=0.99)], {}

    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    studio.segmenter = FakeSegmenter()
    media = studio.root / "assets" / ("c" * 32 + ".png")
    Image.new("RGB", (40, 24), (30, 40, 50)).save(media)
    studio.register_asset(media, "source.png", "image", "c" * 32)
    # Stage 1 must work without a template even in the default replace mode.
    key = studio.create_job({"media_id": "c" * 32})["id"]
    job = studio.jobs[key]
    studio.prepare(job)
    assert job["state"] == "awaiting_selection", job.get("error")
    assert calls == ["license plate", "person"]
    return key, job, calls


def fake_image_render(prepared, template, output, **kwargs):
    from PIL import Image

    # Foreground protection must replay Stage 1 and never call SAM again.
    prepared["private"]["segmenter"].detect(
        Image.fromarray(prepared["private"]["rgb"]), prompt="person", refine=True
    )
    output.mkdir()
    result = output / "result.png"
    Image.new("RGB", (40, 24)).save(result)
    return {
        "output_path": result,
        "timings": {},
        "warnings": [],
        "media_type": "image",
    }


def test_second_stage_reruns_preserve_stage1_and_versioned_results(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    first = job["result"]["output_url"]
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    second = job["result"]["output_url"]
    assert first != second
    assert (studio.root / first.removeprefix("/files/")).is_file()
    assert (studio.root / second.removeprefix("/files/")).is_file()
    assert [r["revision"] for r in job["renders"]] == [1, 2]
    assert all(r["state"] == "completed" for r in job["renders"])
    assert calls == ["license plate", "person"]
    assert job["stage1"]["state"] == "ready"
    assert job["result"]["stage1_reused"] is True


def test_completed_stage1_restores_after_restart_without_loading_sam(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    studio.render(job, 0)

    def forbidden_model_load(*args):
        pytest.fail("A completed Stage 1 must not load SAM during Stage 2")

    restored = Studio(studio.config, segmenter_factory=forbidden_model_load)
    try:
        monkeypatch.setattr(restored.executor, "submit", lambda *a, **k: None)
        pending = restored.jobs[key]
        assert pending["state"] == "completed"
        restored.request_render(key, {"target_id": 0, "mode": "mosaic"})
        restored.render(pending, 0)
        assert pending["state"] == "completed", pending.get("error")
        assert pending["render_revision"] == 2
        assert restored.segmenter is None
        assert calls == ["license plate", "person"]
    finally:
        restored.close()


def test_render_submission_freezes_mode_template_model_and_protection(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, _ = prepared_image_job(studio, monkeypatch)
    studio.config.advisor_profile = "sol-medium"
    media = studio.assets["c" * 32]["_path"]
    studio.register_asset(media, "new-template.png", "image", "d" * 32)
    studio.request_render(
        key,
        {
            "target_id": 0,
            "mode": "replace",
            "template_id": "d" * 32,
            "ai_assist": True,
            "ai_model": "gpt-6-astra",
            "ai_effort": "xhigh",
            "protect_foreground": False,
        },
    )
    with pytest.raises(ValueError, match="仍在运行"):
        studio.request_render(key, {"target_id": 0})
    job.update(
        mode="mosaic", ai_model="gpt-6-luna", ai_effort="low", protect_foreground=True
    )

    def frozen_render(p, template, output, **kwargs):
        assert template == media
        assert kwargs["mode"] == "replace"
        assert kwargs["advisor_profile"] == "astra-xhigh"
        assert kwargs["protect_foreground"] is False
        return fake_image_render(p, template, output, **kwargs)

    monkeypatch.setattr("apexveil.workflows.image.render_image", frozen_render)
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert job["result"]["render_config"]["mode"] == "replace"
    assert job["result"]["ai_selection"]["model"] == "gpt-6-astra"


def test_changed_source_or_interval_cannot_reuse_stage1(running_studio, monkeypatch):
    studio, _ = running_studio
    key, job, _ = prepared_image_job(studio, monkeypatch)
    for changed in ({"media_id": "a" * 32}, {"start": 1}, {"duration": 3}):
        with pytest.raises(ValueError, match="素材或片段已变化"):
            studio.request_render(key, {"target_id": 0, "mode": "blur", **changed})
    assert job["state"] == "awaiting_selection"
    job["_media"]["_path"].write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="源素材已变化"):
        studio.request_render(key, {"target_id": 0, "mode": "blur"})
    assert job["stage1"]["state"] == "invalid"


def test_stage2_failure_preserves_cache_and_previous_download(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    studio.render(job, 0)
    first = job["result"]["output_url"]

    def fail(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr("apexveil.workflows.image.render_image", fail)
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    studio.render(job, 0)
    assert job["state"] == "failed"
    assert job["result"]["output_url"] == first
    assert job["stage1"]["state"] == "ready"
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert calls == ["license plate", "person"]


def test_video_stage2_reruns_and_restart_rehydrate_without_tracking(
    running_studio, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from test_video_workflow import fixture

    studio, _ = running_studio
    source, _, predictor = fixture(tmp_path)
    asset = studio.root / "assets" / ("e" * 32 + ".mkv")
    source.rename(asset)
    studio.register_asset(asset, "clip.mkv", "video", "e" * 32)
    monkeypatch.setattr(studio.executor, "submit", lambda *a, **k: None)
    studio.segmenter = SimpleNamespace(predictor=predictor)
    key = studio.create_job({"media_id": "e" * 32, "mode": "blur", "duration": 0.4})[
        "id"
    ]
    job = studio.jobs[key]
    studio.prepare(job)
    assert job["state"] == "awaiting_selection", job.get("error")
    tracking_calls = len(predictor.requests)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    first = job["result"]["output_url"]
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert len(predictor.requests) == tracking_calls
    assert job["result"]["output_url"] != first
    assert (studio.root / first.removeprefix("/files/")).is_file()
    restored = Studio(studio.config)
    try:
        monkeypatch.setattr(restored.executor, "submit", lambda *a, **k: None)
        resumed = restored.jobs[key]
        restored.request_render(key, {"target_id": 0, "mode": "blur"})
        restored.render(resumed, 0)
        assert resumed["state"] == "completed", resumed.get("error")
        assert resumed["render_revision"] == 3
        assert restored.segmenter is None
        assert len(predictor.requests) == tracking_calls
        with pytest.raises(ValueError, match="素材或片段已变化"):
            restored.request_render(key, {"target_id": 0, "start": 0.1})
        assert resumed["stage1"]["state"] == "ready"
    finally:
        restored.close()


def test_legacy_image_adds_only_missing_person_cache_once(running_studio, monkeypatch):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    folder = job["_directory"] / "prepare"
    plate_before = (folder / "native-masks/plate-0.png").read_bytes()
    (folder / "foreground-cache.json").unlink()
    job["stage1"].pop("cache_version")
    studio.release_prepared(job)
    studio.config.default_template = job["_media"]["_path"]
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(
        key, {"target_id": 0, "mode": "replace", "protect_foreground": True}
    )
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert job["stage1"]["foreground_upgraded"] is True
    assert calls == ["license plate", "person", "person"]
    assert (folder / "native-masks/plate-0.png").read_bytes() == plate_before
    studio.request_render(key, {"target_id": 0})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert calls == ["license plate", "person", "person"]


def test_restart_interrupts_stage2_without_losing_stage1_or_prior_results(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, _ = prepared_image_job(studio, monkeypatch)
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    studio.render(job, 0)
    first = job["result"]["output_url"]
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    # Queued worker was not executed before process shutdown.
    restored = Studio(studio.config)
    try:
        monkeypatch.setattr(restored.executor, "submit", lambda *a, **k: None)
        pending = restored.jobs[key]
        assert pending["state"] == "awaiting_selection"
        assert pending["renders"][-1]["state"] == "interrupted"
        assert pending["result"]["output_url"] == first
        restored.request_render(key, {"target_id": 0})
        restored.render(pending, 0)
        assert pending["state"] == "completed", pending.get("error")
        assert pending["render_revision"] == 3
        assert restored.segmenter is None
    finally:
        restored.close()


def test_legacy_completed_result_without_render_id_can_regenerate(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    monkeypatch.setattr("apexveil.workflows.image.render_image", fake_image_render)
    # Published pre-stage jobs have a result but no version records.
    job.update(
        state="completed",
        mode="replace",
        result={"output_url": "/files/old-result.png"},
    )
    studio.request_render(key, {"target_id": 0, "mode": "mosaic"})
    studio.render(job, 0)
    assert job["state"] == "completed", job.get("error")
    assert job["renders"][0]["mode"] == "replace"
    assert "render_id" not in job["renders"][0]
    assert job["renders"][-1]["mode"] == "mosaic"
    assert calls == ["license plate", "person"]


def test_worker_setup_failure_is_terminal_and_keeps_stage1(running_studio, monkeypatch):
    studio, _ = running_studio
    key, job, calls = prepared_image_job(studio, monkeypatch)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})

    def broken_estimate(*args):
        raise RuntimeError("duration history unavailable")

    monkeypatch.setattr(studio, "expected_seconds", broken_estimate)
    studio.render(job, 0)
    assert job["state"] == "failed"
    assert "duration history unavailable" in job["error"]
    assert job["renders"][-1]["state"] == "failed"
    assert job["stage1"]["state"] == "ready"
    assert calls == ["license plate", "person"]


def test_http_snapshot_cannot_change_during_later_render_mutations(
    running_studio, monkeypatch
):
    studio, _ = running_studio
    key, job, _ = prepared_image_job(studio, monkeypatch)
    snapshot = studio.snapshot(key)
    studio.request_render(key, {"target_id": 0, "mode": "blur"})
    assert snapshot["state"] == "awaiting_selection"
    assert snapshot["renders"] == []
    assert len(studio.snapshot(key)["renders"]) == 1
