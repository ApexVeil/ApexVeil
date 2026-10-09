"""SAM3.1 integration using the public upstream predictor API."""

import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def build_predictor(settings):
    if not settings.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint missing. Run apexveil models download first.")
    from sam3.model_builder import build_sam3_predictor
    return build_sam3_predictor(checkpoint_path=str(settings.checkpoint), version="sam3.1",
                               use_fa3=settings.use_fa3, compile=settings.compile,
                               warm_up=False, async_loading_frames=False)


def summarize_output(response: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = response["outputs"]
    def array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    ids = array(outputs.get("out_obj_ids", [])).reshape(-1)
    masks = array(outputs.get("out_binary_masks", []))
    records = []
    for index, object_id in enumerate(ids):
        mask = masks[index].squeeze().astype(bool)
        name = f"object_{int(object_id):04d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / name)
        records.append({"object_id": int(object_id), "mask": name, "visible_pixels": int(mask.sum())})
    return {"frame_index": int(response["frame_index"]), "objects": records}


def smoke(settings, resource: Path, output: Path, text: str, max_frames: int) -> dict:
    """Validate loading and propagation; this does not replace pixels."""
    from ..models import model_assets
    import torch
    integrity = model_assets(settings)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    predictor = build_predictor(settings)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    session_id = None
    records = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        session = predictor.handle_request({"type": "start_session", "resource_path": str(resource.resolve()),
                                            "offload_video_to_cpu": True})
        session_id = session["session_id"]
        started = time.perf_counter()
        response = predictor.handle_request({"type": "add_prompt", "session_id": session_id,
                                             "frame_index": 0, "text": text,
                                             "output_prob_thresh": settings.confidence_threshold})
        records.append(summarize_output(response, output / "frame_000000"))
        if max_frames > 1:
            stream = predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id,
                                                       "propagation_direction": "forward", "start_frame_index": 0,
                                                       "max_frame_num_to_track": max_frames})
            for item in stream:
                idx = int(item["frame_index"])
                records.append(summarize_output(item, output / f"frame_{idx:06d}"))
        torch.cuda.synchronize()
        result = {"kind": "segmentation_smoke_test", "pixel_replacement": False,
                  "model_integrity": integrity, "resource_name": resource.name,
                  "prompt": text, "model_load_seconds": load_seconds,
                  "inference_seconds": time.perf_counter() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "frames": records}
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        if session_id:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
