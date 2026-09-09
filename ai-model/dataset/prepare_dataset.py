import os
import json
import shutil
import random
from pathlib import Path

# Config
DATASET_DIR = Path(__file__).parent
RAW_DATA_DIR = DATASET_DIR / "raw_flir"
PROCESSED_DIR = DATASET_DIR / "processed"
CLASSES_TO_KEEP = {"person": 1, "car": 3, "bicycle": 2} # Based on standard FLIR COCO classes, mapped to our own 0, 1, 2
YOLO_CLASSES = {"person": 0, "car": 1, "bicycle": 2}

def setup_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(PROCESSED_DIR / split / "images", exist_ok=True)
        os.makedirs(PROCESSED_DIR / split / "labels", exist_ok=True)

def convert_coco_to_yolo(coco_json_path, image_dir, split_name, subset_fraction=0.5):
    """
    Reads COCO JSON, filters classes, converts to YOLO format, and copies a subset of images.
    subset_fraction: Only process a fraction of the dataset (e.g. 0.5 for half)
    """
    if not coco_json_path.exists():
        print(f"Warning: {coco_json_path} not found. Please place the dataset in {RAW_DATA_DIR}")
        return

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build category map
    category_map = {}
    for cat in coco_data['categories']:
        if cat['name'] in YOLO_CLASSES:
            category_map[cat['id']] = YOLO_CLASSES[cat['name']]

    # Build image map
    images = coco_data['images']
    
    # TAKE ONLY HALF THE DATASET as requested
    num_to_keep = int(len(images) * subset_fraction)
    images = random.sample(images, num_to_keep)
    print(f"[{split_name}] Selected {len(images)} images ({subset_fraction*100}% subset).")
    
    image_map = {img['id']: img for img in images}

    # Process annotations
    annotations_by_image = {img_id: [] for img_id in image_map.keys()}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id in image_map and ann['category_id'] in category_map:
            # Convert bbox [x, y, w, h] to YOLO [x_center, y_center, w, h] (normalized)
            img = image_map[img_id]
            img_w, img_h = img['width'], img['height']
            x, y, w, h = ann['bbox']
            
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            yolo_class = category_map[ann['category_id']]
            annotations_by_image[img_id].append(f"{yolo_class} {x_center} {y_center} {norm_w} {norm_h}")

    # Copy images and write labels
    for img_id, img_info in image_map.items():
        src_img_path = image_dir / img_info['file_name'].split('/')[-1] # Handle potential subdirs in file_name
        
        # Some versions of FLIR put images in different folders, let's just search if not direct
        if not src_img_path.exists():
            # Try flat structure
            src_img_path = image_dir / os.path.basename(img_info['file_name'])
            if not src_img_path.exists():
                continue # Skip if image missing
                
        base_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
        
        # Copy image
        dst_img_path = PROCESSED_DIR / split_name / "images" / f"{base_name}.jpg"
        shutil.copy2(src_img_path, dst_img_path)
        
        # Write label
        dst_label_path = PROCESSED_DIR / split_name / "labels" / f"{base_name}.txt"
        with open(dst_label_path, 'w') as f:
            if annotations_by_image[img_id]:
                f.write('\n'.join(annotations_by_image[img_id]) + '\n')

def generate_yaml():
    yaml_content = f"""
path: {PROCESSED_DIR.absolute()}
train: train/images
val: val/images
test: test/images

nc: 3
names: ['person', 'car', 'bicycle']
"""
    with open(DATASET_DIR / "flir.yaml", "w") as f:
        f.write(yaml_content.strip())

if __name__ == "__main__":
    setup_dirs()
    print("Preparing FLIR dataset (100% of dataset)...")
    
    # We assume standard FLIR structure inside raw_flir
    # If the user hasn't downloaded it, we'll create some dummy directories so it doesn't crash completely.
    os.makedirs(RAW_DATA_DIR / "train", exist_ok=True)
    os.makedirs(RAW_DATA_DIR / "val", exist_ok=True)
    
    def find_json_and_images(split):
        for p in RAW_DATA_DIR.rglob('*.json'):
            if split in p.parts and 'thermal_annotations' in p.name:
                img_dir = p.parent / 'thermal_8_bit'
                if not img_dir.exists():
                    img_dir = p.parent
                return p, img_dir
        return None, None

    train_json, train_imgs = find_json_and_images('train')
    if not train_json:
        print(f"Could not find train thermal_annotations.json inside {RAW_DATA_DIR}.")
        print("Please download the FLIR ADAS dataset and extract it to raw_flir.")
    else:
        convert_coco_to_yolo(train_json, train_imgs, "train", subset_fraction=1.0)
        
        val_json, val_imgs = find_json_and_images('val')
        if val_json:
            convert_coco_to_yolo(val_json, val_imgs, "val", subset_fraction=1.0)
            print("Validation set processed.")
        
        generate_yaml()
        print(f"Dataset preparation complete. YOLO configuration written to {DATASET_DIR / 'flir.yaml'}")
