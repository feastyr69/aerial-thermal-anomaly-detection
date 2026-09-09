import cv2
import glob
import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent
IMG_DIR = BASE_DIR / "ai-model" / "dataset" / "raw_flir" / "train" / "thermal_8_bit"
OUTPUT_DIR = Path(__file__).parent / "sample_videos"
OUTPUT_FILE = OUTPUT_DIR / "mock_thermal.mp4"

def create_video():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Try to find images
    if not IMG_DIR.exists():
        # Maybe it's nested
        possible_dirs = list((BASE_DIR / "ai-model" / "dataset" / "raw_flir").rglob("thermal_8_bit"))
        if not possible_dirs:
            print(f"Error: Could not find any thermal_8_bit directories in raw_flir to create a video from.")
            return
        img_dir_to_use = possible_dirs[0]
    else:
        img_dir_to_use = IMG_DIR
        
    image_files = sorted(glob.glob(str(img_dir_to_use / "*.jpg")))
    
    if not image_files:
        print(f"Error: No jpg images found in {img_dir_to_use}")
        return
        
    print(f"Found {len(image_files)} images. Creating mock video...")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    height, width, layers = first_img.shape
    
    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    video = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, 5.0, (width, height))
    
    # Write images to video (limit to 500 frames so it's not too huge)
    for img_path in image_files[:500]:
        img = cv2.imread(img_path)
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()
    print(f"Successfully created mock video at {OUTPUT_FILE}!")

if __name__ == "__main__":
    create_video()
