import cv2
import argparse
from ultralytics import YOLO

def test_model(image_path, weights_path="weights/best.pt"):
    print(f"Loading model from {weights_path}...")
    model = YOLO(weights_path)
    
    print(f"Running inference on {image_path}...")
    results = model(image_path)
    
    # The results object contains the image with drawn bounding boxes
    for r in results:
        # Plot the predictions on the image array
        annotated_img = r.plot()
        
        # Display the image
        print("Displaying result... Press any key to close the window.")
        cv2.imshow("YOLOv8 Inference Result", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save the output image as well
        out_path = "inference_result.jpg"
        cv2.imwrite(out_path, annotated_img)
        print(f"Result saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image you want to test")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to model weights")
    args = parser.parse_args()
    
    test_model(args.image, args.weights)
