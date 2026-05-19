import tempfile
from pathlib import Path


def get_tracker_config(script_root):
    tracker_config = Path(script_root) / "botsort.yaml"
    if not tracker_config.exists():
        return "botsort.yaml"

    return _resolve_tracker_model_path(tracker_config, Path(script_root))


def _resolve_tracker_model_path(tracker_config, script_root):
    try:
        lines = tracker_config.read_text(encoding="utf-8").splitlines()
    except Exception:
        return str(tracker_config)

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("model:"):
            continue

        value = stripped.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"")
        if not value or value == "auto" or Path(value).is_absolute():
            return str(tracker_config)

        candidates = [
            script_root / value,
            script_root.parent / "weights" / value,
            script_root.parent / value,
        ]
        model_path = next((path for path in candidates if path.exists()), None)
        if model_path is None:
            return str(tracker_config)

        indent = line[:len(line) - len(line.lstrip())]
        lines[idx] = f"{indent}model: {model_path}"
        resolved_path = Path(tempfile.gettempdir()) / f"{tracker_config.stem}_resolved_{abs(hash(str(model_path))) & 0xffffffff}.yaml"
        resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(resolved_path)

    return str(tracker_config)


def get_result_track_ids(result, count):
    boxes = getattr(result, "boxes", None)
    ids = getattr(boxes, "id", None) if boxes is not None else None
    if ids is None:
        return [None] * count

    try:
        track_ids = [int(track_id) for track_id in ids.cpu().numpy().reshape(-1).tolist()]
        if len(track_ids) < count:
            track_ids.extend([None] * (count - len(track_ids)))
        return track_ids[:count]
    except Exception:
        return [None] * count


def collect_tracked_objects(results):
    tracked_objects = []
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        try:
            xyxy = boxes.xyxy.cpu().numpy()
        except Exception:
            continue

        track_ids = get_result_track_ids(result, xyxy.shape[0])
        try:
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else [None] * xyxy.shape[0]
        except Exception:
            clss = [None] * xyxy.shape[0]

        for idx in range(xyxy.shape[0]):
            track_id = track_ids[idx] if idx < len(track_ids) else None
            if track_id is None:
                continue
            tracked_objects.append({
                "bbox": [float(v) for v in xyxy[idx].tolist()],
                "class_id": int(clss[idx]) if idx < len(clss) and clss[idx] is not None else None,
                "track_id": int(track_id),
            })

    return tracked_objects


def assign_track_ids_by_iou(detections, tracked_objects, bbox_key, id_key="track_id", class_key=None, iou_threshold=0.3):
    used_ids = set()
    for det in detections:
        bbox = det.get(bbox_key)
        if not bbox:
            continue

        det_class = det.get(class_key) if class_key else None
        best_track = None
        best_iou = 0.0
        for tracked in tracked_objects:
            track_id = tracked.get("track_id")
            if track_id is None or track_id in used_ids:
                continue
            if class_key and tracked.get("class_id") is not None and det_class is not None and tracked.get("class_id") != det_class:
                continue

            iou = bbox_iou(bbox, tracked["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_track = tracked

        if best_track is not None and best_iou >= iou_threshold:
            det[id_key] = int(best_track["track_id"])
            used_ids.add(int(best_track["track_id"]))
        else:
            det[id_key] = None

    return detections


def bbox_iou(box1, box2):
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area
