"""Convert a YOLO detection split into a minimal COCO detection annotation file.

The converter never copies images.  COCO ``file_name`` entries stay relative to
the supplied dataset root, so the teacher reads exactly the images used by
YOLO.  It supports the project's two existing layouts:

* flat: ``<root>/<split>/images`` and ``<root>/<split>/labels``;
* nested: ``<root>/images/<split>`` and ``<root>/labels/<split>``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass
class SplitSummary:
    split: str
    images: int = 0
    annotations: int = 0
    empty_images: int = 0
    missing_label_files: int = 0
    malformed_label_lines: int = 0
    skipped_boxes: int = 0


def parse_split(spec: str) -> tuple[str, str]:
    try:
        source, target = spec.split(":", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("split must be SOURCE:OUTPUT") from error
    if not source or not target:
        raise argparse.ArgumentTypeError("split source and output names must be non-empty")
    return source, target


def split_paths(root: Path, source_split: str, layout: str) -> tuple[Path, Path]:
    flat_images = root / source_split / "images"
    flat_labels = root / source_split / "labels"
    nested_images = root / "images" / source_split
    nested_labels = root / "labels" / source_split
    if layout == "flat" or (layout == "auto" and flat_images.is_dir()):
        return flat_images, flat_labels
    if layout == "nested" or (layout == "auto" and nested_images.is_dir()):
        return nested_images, nested_labels
    raise FileNotFoundError(
        f"cannot find images for split '{source_split}' under {root}; expected {flat_images} or {nested_images}"
    )


def parse_label(label_path: Path, width: int, height: int, summary: SplitSummary) -> list[dict]:
    if not label_path.exists():
        summary.missing_label_files += 1
        return []
    annotations = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5:
            summary.malformed_label_lines += 1
            continue
        try:
            class_id = int(fields[0])
            center_x, center_y, box_w, box_h = (float(value) for value in fields[1:])
        except ValueError:
            summary.malformed_label_lines += 1
            continue
        if class_id != 0 or not all(0.0 <= value <= 1.0 for value in (center_x, center_y, box_w, box_h)):
            summary.skipped_boxes += 1
            continue
        pixel_w, pixel_h = box_w * width, box_h * height
        x0, y0 = (center_x * width) - pixel_w / 2, (center_y * height) - pixel_h / 2
        x0, y0 = max(0.0, x0), max(0.0, y0)
        pixel_w, pixel_h = min(pixel_w, width - x0), min(pixel_h, height - y0)
        if pixel_w <= 0.0 or pixel_h <= 0.0:
            summary.skipped_boxes += 1
            continue
        annotations.append(
            {
                "category_id": 1,
                "bbox": [round(x0, 4), round(y0, 4), round(pixel_w, 4), round(pixel_h, 4)],
                "area": round(pixel_w * pixel_h, 4),
                "iscrowd": 0,
                "source_line": line_number,
            }
        )
    return annotations


def convert_split(root: Path, source_split: str, output_path: Path, layout: str) -> SplitSummary:
    image_dir, label_dir = split_paths(root, source_split, layout)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")
    summary = SplitSummary(split=source_split)
    images, annotations = [], []
    annotation_id = 1
    image_id = 1
    for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(image_path) as image:
            width, height = image.size
        split_annotations = parse_label(label_dir / f"{image_path.stem}.txt", width, height, summary)
        relative_name = image_path.relative_to(root).as_posix()
        images.append({"id": image_id, "file_name": relative_name, "width": width, "height": height})
        for annotation in split_annotations:
            annotation.pop("source_line")
            annotation["id"] = annotation_id
            annotation["image_id"] = image_id
            annotations.append(annotation)
            annotation_id += 1
        summary.images += 1
        summary.annotations += len(split_annotations)
        summary.empty_images += int(not split_annotations)
        image_id += 1
    payload = {
        "info": {"description": "YOLO-to-COCO conversion for Grounding DINO teacher training"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "apple", "supercategory": "fruit"}],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", choices=("auto", "flat", "nested"), default="auto")
    parser.add_argument(
        "--split",
        type=parse_split,
        action="append",
        required=True,
        metavar="SOURCE:OUTPUT",
        help="source YOLO split and destination COCO filename stem, e.g. train:train",
    )
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    summaries = []
    for source_split, output_split in args.split:
        summary = convert_split(root, source_split, output_dir / f"{output_split}.json", args.layout)
        summaries.append(asdict(summary))
    report = {"dataset_root": str(root), "layout": args.layout, "splits": summaries}
    (output_dir / "conversion_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
