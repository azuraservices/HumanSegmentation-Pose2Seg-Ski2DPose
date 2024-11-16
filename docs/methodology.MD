# Methodology

This document outlines the methodology followed to reproduce the results of the Pose2Seg model and adapt it to the Ski2DPose dataset.

---

## Overview
The methodology is divided into two main phases:
1. **Reproducing the Results of Pose2Seg**:
   - Clone the official Pose2Seg repository.
   - Install the required dependencies.
   - Train the model on the COCO dataset to verify reproducibility.
2. **Adapting and Testing with Ski2DPose**:
   - Convert the Ski2DPose dataset to the COCO format.
   - Modify the model to handle the new dataset.
   - Evaluate the model's performance qualitatively and quantitatively.

---

## Steps Followed

### 1. Environment Setup
- **Tools**: Python, PyTorch, Google Colab (optional).
- **Dependencies**: Installed using the `requirements.txt` file. Includes:
  - `torch`, `torchvision`, `opencv-python`, `pycocotools`, `Polygon3`.
- **Repository**: Cloned the official Pose2Seg GitHub repository.

Commands used:
```
!git clone https://github.com/liruilong940607/Pose2Seg.git
%cd Pose2Seg
```

---

### 2. Training the Pose2Seg Model
- The model was trained using the COCO dataset for 15 epochs to validate reproducibility.
- Configuration:
  - Optimizer: SGD
  - Learning Rate: 0.001
  - Batch Size: 16
- The model weights were saved in a file named `last.pkl`.

Example command:
```
python src/train.py --config config.json
```

---

### 3. Dataset Adaptation (Ski2DPose to COCO Format)
The Ski2DPose dataset was adapted to the COCO format to be compatible with Pose2Seg. Key steps:
1. **Keypoint Mapping**:
   - Ski2DPose includes additional keypoints (e.g., ski poles). These were mapped or excluded.
2. **Bounding Box and Mask Creation**:
   - Used the `Polygon` library to generate ground truth segmentation masks.
3. **JSON Conversion**:
   - Saved all annotations in a COCO-compliant JSON file.

Scripts used:
```
python src/adapt_dataset.py --input data/ski2dpose --output data/coco_format
```

---

### 4. Testing and Evaluation
- The adapted model was tested on Ski2DPose to evaluate its performance.
- The evaluation included:
  - **Qualitative Analysis**: Comparing predicted masks with ground truth.
  - **Quantitative Metrics**: Calculating precision (AP) and recall (AR) for various object sizes.

Example command:
```
python src/test.py --weights results/last.pkl --dataset ski2dpose
```

---

## Results
- **COCO Dataset**:
  - Achieved performance close to the original paper's results (AP = 0.564 vs. 0.582 reported).
- **Ski2DPose Dataset**:
  - Predicted masks were visually accurate, capturing skier shapes but with minor errors in occluded areas.

---

## Challenges
- Limited GPU resources constrained the training duration to 15 epochs.
- Adapting Ski2DPose to COCO format required significant customization of annotations and masks.

---

## Future Work
1. **Extended Training**:
   - Train the model for more epochs to improve performance.
2. **Alternative Datasets**:
   - Test the model on other datasets to evaluate generalization.
3. **Automated Comparison**:
   - Use benchmarking tools to compare Pose2Seg with other segmentation models.

---

## Conclusion
The methodology effectively reproduced the results of Pose2Seg and demonstrated its adaptability to the Ski2DPose dataset. While qualitative results were promising, further optimization is required for better performance in complex scenarios.
