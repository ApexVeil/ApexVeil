"""SAM3.1 integration using the public upstream predictor API."""

import json
import inspect
from functools import wraps
from pathlib import Path
import time
import numpy as np
from PIL import Image



def install_interactive_output_compat(model):
    """Keep freshly tracked point-prompt masks when no VG cache exists.

    The pinned multiplex adapter returns early for an uncached frame before
    merging the tracker masks supplied in that same call. A fresh point-only
    session therefore drops valid predictions. Repair output assembly on this
    instance; do not change inference, scores, visibility, or action history.
    """
    if getattr(model, "apexveil_interactive_output_compat", False):
        return
    original = model._build_sam2_output

    @wraps(original)
    def compatible_output(inference_state, frame_idx, refined_obj_id_to_mask=None):
        result = original(inference_state, frame_idx, refined_obj_id_to_mask)
        if frame_idx not in inference_state["cached_frame_outputs"] and refined_obj_id_to_mask:
            result = dict(result)
            result.update(refined_obj_id_to_mask)
        return result

    model._build_sam2_output = compatible_output
    model.apexveil_interactive_output_compat = True


def build_predictor(settings):
    if not settings.checkpoint.is_file():
        raise FileNotFoundError("Checkpoint missing. Run apexveil models download first.")
    from sam3.model_builder import build_sam3_predictor
    predictor = build_sam3_predictor(
        checkpoint_path=str(settings.checkpoint), version="sam3.1",
        use_fa3=settings.use_fa3, use_rope_real=False, compile=settings.compile,
        warm_up=False, async_loading_frames=False,
    )
    # The pinned shared predictor forwards this optional argument, while its
    # multiplex model does not accept it. Adapt this instance only; reject a
    # request to enable unsupported state offload instead of ignoring it.
    original_init = predictor.model.init_state
    if "offload_state_to_cpu" not in inspect.signature(original_init).parameters:
        @wraps(original_init)
        def compatible_init(*args, **kwargs):
            if kwargs.pop("offload_state_to_cpu", False):
                raise NotImplementedError("SAM3.1 multiplex does not support state offload")
            return original_init(*args, **kwargs)
        predictor.model.init_state = compatible_init
    # Upstream loads the assembled model with strict=False. Check every learned
    # tensor explicitly so a superficially successful load cannot go unnoticed.
    import torch
    state = torch.load(settings.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    state = {key.replace("sam3_model.", "detector.", 1) if key.startswith("sam3_model.")
             else key.replace("sam2_predictor.", "tracker.", 1) if key.startswith("sam2_predictor.")
             else key: value for key, value in state.items()}
    parameters = dict(predictor.model.named_parameters())
    missing = sorted(set(parameters) - set(state))
    mismatched = [key for key, value in parameters.items()
                  if key in state and value.shape != state[key].shape]
    if missing or mismatched:
        raise RuntimeError(f"Incomplete learned checkpoint: missing={missing[:8]}, shape={mismatched[:8]}")
    install_interactive_output_compat(predictor.model)
    predictor.apexveil_load_audit = {"interactive_output_compat": "merge_fresh_tracker_masks_without_vg_cache",
                                   "learned_tensors_checked": len(parameters),
                                   "missing_learned_tensors": 0, "shape_mismatches": 0}
    del state
    return predictor


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
                  "model_integrity": integrity, "load_audit": predictor.apexveil_load_audit, "resource_name": resource.name,
                  "prompt": text, "model_load_seconds": load_seconds,
                  "inference_seconds": time.perf_counter() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "frames": records}
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        if session_id:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
