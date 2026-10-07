"""Convert the standard HIT-UAV COCO annotations to Ultralytics YOLO format.

Expected source: the official HIT-UAV archive (or a faithful Kaggle copy) with
normal_json/{train,val,test}.json and the matching split image folders.
"""

import argparse
import json
import math
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("person", "vehicle")
CLASS_ALIASES = {
    "person": 0,
    "car": 1,
    "bicycle": 1,
    "othervehicle": 1,
    "othervechicle": 1,
    "other vehicle": 1,
    "other_vehicle": 1,
}


def _find_annotation_file(raw_dir: Path, split: str) -> Path:
    matches = [
        path
        for path in raw_dir.rglob(f"{split}.json")
        if path.parent.name.lower() == "normal_json"
    ]
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            f"Expected one normal_json/{split}.json under {raw_dir}; found: {found}"
        )
    return matches[0]


def _resolve_image(raw_dir: Path, annotation_file: Path, split: str, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (
        raw_dir / relative,
        annotation_file.parent / relative,
        annotation_file.parent / split / relative.name,
        annotation_file.parent.parent / split / relative.name,
        raw_dir / "normal_json" / split / relative.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = list(raw_dir.rglob(relative.name))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred = [path for path in matches if "normal_json" in path.parts and split in path.parts]
        if len(preferred) == 1:
            return preferred[0]
    raise FileNotFoundError(f"Could not uniquely locate image {file_name!r} for {split} split")


def _convert_split(raw_dir: Path, output_dir: Path, split: str) -> tuple[int, int]:
    annotation_file = _find_annotation_file(raw_dir, split)
    with annotation_file.open(encoding="utf-8") as source:
        coco = json.load(source)

    category_map = {}
    for category in coco.get("categories", []):
        normalized_name = " ".join(category["name"].casefold().replace("_", " ").split())
        class_id = CLASS_ALIASES.get(normalized_name)
        if class_id is not None:
            category_map[category["id"]] = class_id

    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image = {image_id: [] for image_id in images_by_id}
    for annotation in coco.get("annotations", []):
        class_id = category_map.get(annotation.get("category_id"))
        image_id = annotation.get("image_id")
        if class_id is None or image_id not in annotations_by_image:
            continue
        bbox = annotation.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, width, height = map(float, bbox)
        image = images_by_id[image_id]
        image_width, image_height = float(image["width"]), float(image["height"])
        if not all(map(math.isfinite, (x, y, width, height, image_width, image_height))):
            continue
        if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
            continue

        center_x = min(1.0, max(0.0, (x + width / 2) / image_width))
        center_y = min(1.0, max(0.0, (y + height / 2) / image_height))
        normalized_width = min(1.0, max(0.0, width / image_width))
        normalized_height = min(1.0, max(0.0, height / image_height))
        if normalized_width == 0 or normalized_height == 0:
            continue
        annotations_by_image[image_id].append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )

    image_dir = output_dir / split / "images"
    label_dir = output_dir / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image in coco.get("images", []):
        source_image = _resolve_image(raw_dir, annotation_file, split, image["file_name"])
        destination_image = image_dir / Path(image["file_name"]).name
        destination_label = label_dir / f"{destination_image.stem}.txt"
        if destination_image.exists() or destination_label.exists():
            raise FileExistsError(
                f"Output already contains {destination_image.name}; clear {output_dir} before rerunning"
            )
        shutil.copy2(source_image, destination_image)
        destination_label.write_text(
            "\n".join(annotations_by_image[image["id"]]) + "\n"
            if annotations_by_image[image["id"]]
            else "",
            encoding="utf-8",
        )
        copied += 1

    box_count = sum(len(lines) for lines in annotations_by_image.values())
    return copied, box_count


def prepare_dataset(raw_dir: Path, output_dir: Path) -> Path:
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"HIT-UAV source directory does not exist: {raw_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in SPLITS:
        split_counts[split] = _convert_split(raw_dir, output_dir, split)

    yaml_path = output_dir / "hit_uav.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(str(output_dir))}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {json.dumps(list(CLASS_NAMES))}\n",
        encoding="utf-8",
    )
    for split, (image_count, box_count) in split_counts.items():
        print(f"{split}: {image_count} images, {box_count} target boxes")
    print(f"Wrote dataset config: {yaml_path}")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).parent / "raw_hit_uav",
        help="Extracted HIT-UAV archive directory (default: dataset/raw_hit_uav)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "processed_hit_uav",
        help="YOLO-format output directory (default: dataset/processed_hit_uav)",
    )
    args = parser.parse_args()
    prepare_dataset(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
