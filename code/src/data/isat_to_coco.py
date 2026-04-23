import argparse
import json
import random
import re
from pathlib import Path

from ..utils.io import write_json
from ..utils.paths import ensure_dir


def flatten_segmentation(points: list[list[float]]) -> list[float]:
    return [coord for pair in points for coord in pair]


def load_categories(categories_path: str) -> tuple[list[dict], dict[str, int]]:
    path = Path(categories_path)
    if not path.exists():
        raise FileNotFoundError(f"Categories file not found: {path}")

    names: list[str] = []

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "label" in payload:
            names = [item.get("name", "") for item in payload.get("label", []) if isinstance(item, dict)]
        elif isinstance(payload, list):
            names = [str(item) for item in payload]
    else:
        # Minimal parser for simple ISAT yaml style lines such as: "- name: tomato"
        text = path.read_text(encoding="utf-8")
        names = re.findall(r"name\s*:\s*['\"]?([^'\"\n\r]+)", text)

    names = [n.strip() for n in names if n and n.strip() and n.strip() != "__background__"]

    if not names:
        raise ValueError(
            f"No usable categories parsed from {path}. "
            "Expected ISAT label entries with 'name' fields (excluding __background__)."
        )

    categories: list[dict] = []
    cat_map: dict[str, int] = {}
    for idx, name in enumerate(names, start=1):
        categories.append({"id": idx, "name": name, "supercategory": "none"})
        cat_map[name] = idx

    return categories, cat_map


def convert_isat_folder_to_coco(
    task_dir: str,
    label_dir: str,
    categories_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 888,
    selected_labels: list[str] | None = None,
) -> dict:
    img_dir = Path(task_dir)
    lbl_dir = Path(label_dir)
    out_dir = ensure_dir(output_dir)

    if not img_dir.exists():
        raise FileNotFoundError(f"Task image directory not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    categories, category_map = load_categories(categories_path)

    # Optional subset filtering by class name
    if selected_labels is not None:
        valid_set = set(selected_labels)
        categories = [c for c in categories if c["name"] in valid_set]
        category_map = {k: v for k, v in category_map.items() if k in valid_set}

    image_files = sorted([p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    json_map = {p.stem: p.name for p in sorted(lbl_dir.glob("*.json"))}

    dataset = []
    unlabeled_images = []
    for img_name in image_files:
        stem = Path(img_name).stem
        if stem in json_map:
            json_path = lbl_dir / json_map[stem]
            with open(json_path, "r", encoding="utf-8") as f:
                isat = json.load(f)
            
            has_valid = False
            for obj in isat.get("objects", []):
                if obj.get("category") in category_map:
                    has_valid = True
                    break
            
            if has_valid:
                dataset.append({"img_file": img_name, "json_file": json_map[stem]})
            else:
                unlabeled_images.append(img_name)
        else:
            unlabeled_images.append(img_name)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    paths = {
        "train": dataset[:n_train],
        "val": dataset[n_train:n_val],
        "test": dataset[n_val:],
    }

    for split_name, split_data in paths.items():
        coco = {"images": [], "annotations": [], "categories": categories}
        ann_id = 1
        img_id = 1

        for item in split_data:
            json_path = lbl_dir / item["json_file"]
            with open(json_path, "r", encoding="utf-8") as f:
                isat = json.load(f)

            info = isat.get("info", {})
            coco["images"].append(
                {
                    "file_name": item["img_file"],
                    "id": img_id,
                    "width": int(info.get("width", 0)),
                    "height": int(info.get("height", 0)),
                }
            )

            for obj in isat.get("objects", []):
                cat = obj.get("category")
                if cat not in category_map:
                    continue

                seg_flat = flatten_segmentation(obj.get("segmentation", []))
                if len(seg_flat) < 6:
                    continue

                bbox = obj.get("bbox", [0, 0, 0, 0])
                area = obj.get("area", bbox[2] * bbox[3] if len(bbox) >= 4 else 0)

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_map[cat],
                        "segmentation": [seg_flat],
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": int(obj.get("iscrowd", 0)),
                        "group_id": obj.get("group", None),
                    }
                )
                ann_id += 1

            img_id += 1

        write_json(out_dir / f"{split_name}.json", coco)

    # Write unlabeled images to a separate json list file
    write_json(out_dir / "unlabeled_images.json", unlabeled_images)

    return {
        "total_pairs": n_total,
        "split_counts": {
            "train": n_train,
            "val": n_val,
            "test": len(paths["test"]),
        },
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "output_dir": str(out_dir),
        "unlabeled_images": unlabeled_images,
        "counts": {
            "train": n_train,
            "val": n_val,
            "test": len(paths["test"]),
            "total_labeled": n_total,
            "total_unlabeled": len(unlabeled_images),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ISAT annotations to COCO split json files.")
    parser.add_argument("--task-dir", required=True, help="Directory with image files")
    parser.add_argument("--label-dir", required=True, help="Directory with ISAT json labels")
    parser.add_argument("--categories", required=True, help="Path to categories file (isat.yaml or json)")
    parser.add_argument("--output-dir", required=True, help="Output directory for train/val/test json")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=888)
    args = parser.parse_args()

    result = convert_isat_folder_to_coco(
        task_dir=args.task_dir,
        label_dir=args.label_dir,
        categories_path=args.categories,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

