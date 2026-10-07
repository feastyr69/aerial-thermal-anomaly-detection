import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "uavbest.pt"

def test_model(image_path, weights_path=DEFAULT_WEIGHTS, output_path="inference_result.jpg", show=False):
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    
    for r in results:
        annotated_img = r.plot()

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), annotated_img):
            raise RuntimeError(f"Could not save inference result to {out_path}")
        print(f"Result saved to {out_path}")

        if show:
            print("Displaying result... Press any key to close the window.")
            cv2.imshow("YOLOv8 Inference Result", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image you want to test")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--output", type=Path, default=Path("inference_result.jpg"), help="Where to save the annotated image")
    parser.add_argument("--show", action="store_true", help="Display the result in a window (requires a GUI)")
    args = parser.parse_args()
    
    test_model(args.image, args.weights, args.output, args.show)
