import os
import json
import cv2
import numpy as np
from pathlib import Path

# Config
DATASET_DIR = Path(__file__).parent
RAW_DATA_DIR = DATASET_DIR / "raw_flir"

def create_mock_flir():
    print("Generating mock FLIR dataset for testing...")
    
    for split in ["train", "val"]:
        img_dir = RAW_DATA_DIR / split / "thermal_8_bit"
        os.makedirs(img_dir, exist_ok=True)
        
        coco_data = {
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
                {"id": 3, "name": "car"},
                {"id": 4, "name": "dog"}
            ],
            "images": [],
            "annotations": []
        }
        
        # Create dummy images
        num_images = 200 if split == "train" else 40
        for i in range(num_images):
            img_id = i + 1
            file_name = f"dummy_thermal_{i}.jpg"
            img_path = img_dir / file_name
            
            # Generate a noisy grayscale image resembling thermal
            img = np.random.randint(50, 100, (480, 640), dtype=np.uint8)
            
            # Add a "hot" person or car blob
            if i % 2 == 0:
                # Car-like blob
                cv2.rectangle(img, (200, 200), (400, 300), (200,), -1)
                cat_id = 3
                bbox = [200, 200, 200, 100]
            else:
                # Person-like blob
                cv2.rectangle(img, (300, 150), (350, 350), (255,), -1)
                cat_id = 1
                bbox = [300, 150, 50, 200]
                
            # Add some Gaussian blur to look like thermal
            img = cv2.GaussianBlur(img, (15, 15), 0)
            
            cv2.imwrite(str(img_path), img)
            
            coco_data["images"].append({
                "id": img_id,
                "file_name": file_name,
                "width": 640,
                "height": 480
            })
            
            coco_data["annotations"].append({
                "id": img_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": bbox
            })
            
        json_path = RAW_DATA_DIR / split / "thermal_annotations.json"
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        print(f"Generated {split} split with {num_images} dummy images.")

if __name__ == "__main__":
    create_mock_flir()
    print("Mock dataset generated. You can now run prepare_dataset.py successfully.")
