import os
import json
from Polygon import Polygon
import numpy as np
import cv2

def convert_to_coco_format(input_path, output_path):
    """
    Convert the Ski2DPose dataset to COCO format.
    """
    print("Converting Ski2DPose to COCO format...")

    # Initialize COCO structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{
            "id": 1,
            "name": "person",
            "supercategory": "human"
        }]
    }

    # Counters
    annotation_id = 1
    image_id = 1

    # Process each image and annotation in the Ski2DPose dataset
    for file_name in os.listdir(input_path):
        if file_name.endswith(".json"):  # Process JSON files
            with open(os.path.join(input_path, file_name), 'r') as f:
                data = json.load(f)

            # Load image information
            image_file = data["image_path"]
            height, width, _ = cv2.imread(os.path.join(input_path, image_file)).shape

            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "file_name": image_file,
                "width": width,
                "height": height
            })

            # Add annotations
            for person in data["annotations"]:
                keypoints = person["keypoints"]
                segment = person["segment"]
                
                # Convert keypoints to COCO format
                keypoints_coco = []
                for i in range(0, len(keypoints), 3):
                    keypoints_coco.extend([keypoints[i], keypoints[i+1], keypoints[i+2]])

                # Create a bounding box
                x_coordinates = [keypoints[i] for i in range(0, len(keypoints), 3)]
                y_coordinates = [keypoints[i+1] for i in range(0, len(keypoints), 3)]
                x_min, y_min = min(x_coordinates), min(y_coordinates)
                width_bbox, height_bbox = max(x_coordinates) - x_min, max(y_coordinates) - y_min

                # Add annotation entry
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segment],
                    "area": Polygon(segment).area(),
                    "bbox": [x_min, y_min, width_bbox, height_bbox],
                    "iscrowd": 0,
                    "keypoints": keypoints_coco,
                    "num_keypoints": len(keypoints) // 3
                })
                annotation_id += 1

            image_id += 1

    # Save COCO formatted JSON
    output_file = os.path.join(output_path, "ski2dpose_coco.json")
    with open(output_file, 'w') as out_file:
        json.dump(coco_data, out_file)

    print(f"Conversion complete! COCO JSON saved at {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Ski2DPose dataset to COCO format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the Ski2DPose dataset folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the COCO formatted dataset.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    convert_to_coco_format(args.input, args.output)
