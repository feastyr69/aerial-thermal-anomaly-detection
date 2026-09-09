import os
import cv2
import glob
from pathlib import Path

# Config
DATASET_DIR = Path(__file__).parent
PROCESSED_DIR = DATASET_DIR / "processed"
YOLO_CLASSES = {0: "person", 1: "car", 2: "bicycle"}

def print_stats(split):
    label_dir = PROCESSED_DIR / split / "labels"
    image_dir = PROCESSED_DIR / split / "images"
    
    if not label_dir.exists():
        print(f"Split '{split}' not found at {label_dir}")
        return
        
    label_files = glob.glob(str(label_dir / "*.txt"))
    image_files = glob.glob(str(image_dir / "*.jpg"))
    
    print(f"\n--- Stats for {split} split ---")
    print(f"Total images: {len(image_files)}")
    print(f"Total labeled images: {len(label_files)}")
    
    class_counts = {0: 0, 1: 0, 2: 0}
    total_boxes = 0
    
    for lf in label_files:
        with open(lf, 'r') as f:
            lines = f.readlines()
            total_boxes += len(lines)
            for line in lines:
                cls_id = int(line.split(' ')[0])
                if cls_id in class_counts:
                    class_counts[cls_id] += 1
                    
    print(f"Total bounding boxes: {total_boxes}")
    for cls_id, count in class_counts.items():
        print(f"  {YOLO_CLASSES[cls_id]}: {count}")

def visualize_sample(split, num_samples=3):
    image_dir = PROCESSED_DIR / split / "images"
    label_dir = PROCESSED_DIR / split / "labels"
    
    if not image_dir.exists():
        return
        
    image_files = glob.glob(str(image_dir / "*.jpg"))
    if not image_files:
        return
        
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    output_dir = DATASET_DIR / "sample_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {len(samples)} sample visualizations in {output_dir}/ ...")
    
    for i, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = label_dir / f"{base_name}.txt"
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split(' ')
                    cls_id = int(parts[0])
                    x_c, y_c, bw, bh = map(float, parts[1:])
                    
                    # Convert YOLO format back to pixel coordinates
                    x1 = int((x_c - bw/2) * w)
                    y1 = int((y_c - bh/2) * h)
                    x2 = int((x_c + bw/2) * w)
                    y2 = int((y_c + bh/2) * h)
                    
                    color = (0, 255, 0) if cls_id == 0 else (255, 0, 0) if cls_id == 1 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, YOLO_CLASSES[cls_id], (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out_path = output_dir / f"sample_{i}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    print_stats("train")
    print_stats("val")
    visualize_sample("train")
