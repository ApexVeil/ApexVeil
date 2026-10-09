"""Independent SAM3.1 detector adapter with full-frame + crop refinement."""

from dataclasses import dataclass
import time
import numpy as np
from PIL import Image
from .sam31 import build_predictor


@dataclass
class Detection:
    object_id: int
    score: float
    visible_mask: np.ndarray
    diagnostics: dict


class ImageSegmenter:
    def __init__(self, settings):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor

        start = time.perf_counter()
        self.predictor = build_predictor(settings)
        self.processor = Sam3Processor(
            self.predictor.model.detector,
            confidence_threshold=settings.confidence_threshold,
        )
        torch.cuda.synchronize()
        self.metadata = {
            "backend": "sam3.1_detector",
            "load_seconds": time.perf_counter() - start,
            "load_audit": self.predictor.apexveil_load_audit,
            "device": torch.cuda.get_device_name(),
            "threshold": settings.confidence_threshold,
            "mask_holes_filled": False,
        }

    def _predict(self, image, prompt):
        state = self.processor.set_image(image)
        result = self.processor.set_text_prompt(prompt=prompt, state=state)
        width, height = image.size
        masks = (
            result["masks"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1, height, width)
            .astype(np.uint8)
        )
        boxes = result["boxes"].detach().float().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        return masks, boxes, scores

    def detect(self, image: Image.Image, prompt="license plate", refine=True):
        import torch

        width, height = image.size
        started = time.perf_counter()
        detections = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, boxes, scores = self._predict(image, prompt)
            # Image-local IDs are ordered by confidence, never claimed as video IDs.
            order = np.argsort(-scores)
            for i in order[:32]:
                raw, box, score = masks[i], boxes[i], scores[i]
                if raw.sum() < 4:
                    continue
                diagnostics = {"initial_box_xyxy": box.tolist(), "refined": False}
                chosen = raw
                if refine:
                    bw, bh = box[2] - box[0], box[3] - box[1]
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    x0, x1 = max(0, int(cx - bw)), min(width, int(cx + bw))
                    y0, y1 = (
                        max(0, int(cy - max(bh * 1.8, bw * 0.4))),
                        min(height, int(cy + max(bh * 1.8, bw * 0.4))),
                    )
                    if x1 > x0 and y1 > y0:
                        candidates, _, crop_scores = self._predict(
                            image.crop((x0, y0, x1, y1)), prompt
                        )
                        base = raw[y0:y1, x0:x1] > 0
                        if len(candidates):
                            ious = [
                                float(
                                    ((c > 0) & base).sum()
                                    / max(1, ((c > 0) | base).sum())
                                )
                                for c in candidates
                            ]
                            best = int(np.argmax(ious))
                            if ious[best] > 0.25:
                                chosen = np.zeros_like(raw)
                                chosen[y0:y1, x0:x1] = candidates[best]
                                diagnostics.update(
                                    refined=True,
                                    refinement_iou=ious[best],
                                    refinement_score=float(crop_scores[best]),
                                )
                duplicate = any(
                    np.logical_and(d.visible_mask, chosen).sum()
                    / max(1, np.logical_or(d.visible_mask, chosen).sum())
                    > 0.75
                    for d in detections
                )
                if not duplicate:
                    detections.append(
                        Detection(len(detections), float(score), chosen, diagnostics)
                    )
        torch.cuda.synchronize()
        return detections, {
            "segmentation_seconds": time.perf_counter() - started,
            "candidates": len(detections),
            "prompt": prompt,
            "candidate_limit_reached": len(masks) > 32,
        }
