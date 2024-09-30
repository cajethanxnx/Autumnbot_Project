# Autumnbot_Project
### Project Overview
The Autumnbot Leaf Detection project aims to automate the process of detecting fallen autumn leaves using YOLOv8, a state-of-the-art object detection algorithm. This project includes data preprocessing, model training, validation, and performance evaluation, with outputs aimed at integrating into the Autumnbot robotics platform.

### Training Setup
YOLOv8 for object detection was trained using a custom dataset of autumn leaves. The training was conducted using Google Colab to leverage GPU resources for faster processing.

### Training Details
**Training Platform**: Google Colab

**Model**: YOLOv8 (Small configuration)

**Dataset**: Custom dataset with 2076 images annotated in YOLOv5 format

**Training Epochs**: 100

**Image Size**: 640x640

**Batch Size**: 16

**Optimizer**: AdamW optimizer with automatic mixed precision (AMP)

**Learning Rate**: 0.002

### Results Overview
After 100 epochs of training, the model yielded the following performance metrics:

**Box Loss**: Reduction from ~2.0 to ~1.396 by epoch 78.
**Class Loss**: Reduction from ~3.022 to ~1.931 by epoch 78
**DFL Loss**: Reduction from ~2.131 to ~1.646 at epoch 78
**Precision**: 0.595
**Recall**: 0.321
**Mean Average Precision (mAP)@0.5**: 0.339
**Mean Average Precision (mAP)@(50-95)**: 0.163

### Insights from Results
#### Loss Curves
- Loss curves for bounding box, classification, and DFL losses consistently showed a decrease, indicating the model's learning stability over the training epochs.
Make sure you have cloned the project repository where all the required files reside.

#### Precision- Recall F1-Confidence Curve
- The precision-recall curve indicates how the model trades off between precision and recall, with the F1-confidence curve showing the confidence level at which the model achieves its best F1 score.

**F1 Score**: 0.42 at 0.151 confidence
**Precision**: 1.00 at 0.677 confidence

![F1_curve](https://github.com/user-attachments/assets/31417e84-72d9-4a36-bdeb-7ed7645a43ab)

#### Precision-Confidence Curve
- The precision confidence curve highlights that at a confidence threshold of 0.677, the precision for all classes reached 1.00, though precision at lower thresholds shows a variance.

#### Confusion Matrix
- The confusion matrix indicated that 24 instances of autumn leaves were correctly classified, while 63 instances were predicted as background.
- The confusion matrix showed that there were more false negatives than false positives, indicating missed detections. The normalized confusion matrix is shown below:
![confusion_matrix_normalized](https://github.com/user-attachments/assets/60471e59-2199-4f29-bbf4-aa7b3942af05)

### Model Predictions
Predictions made by the model during training and validation showed bounding boxes drawn around detected leaves. The prediction results are visualized in the images below:

![val_batch2_pred](https://github.com/user-attachments/assets/870d201e-1b4f-44e8-b580-993745e631cd)

![val_batch1_pred](https://github.com/user-attachments/assets/c3545b20-0760-41ff-bbb3-370cc226ac6f)

![val_batch0_pred](https://github.com/user-attachments/assets/51d36788-ce67-4388-b68a-a8aa418fbf11)

### Validation Set Results
**Validation Dataset**: 108 images.
**Classes**: Autumn Leaves and Background.
**Performance**: mAP@0.5 of 0.339, indicating moderate success in detecting leaves across the validation set.

### Installation
1. Clone the repository and navigate to the project directory.

2. Install dependencies:

`pip install -r requirements.txt`

3. (Optional) If using Google Colab for training, upload the dataset and set the path in the training script.

### Running the Model
To train the YOLOv8 model, use the following command:
`python train.py --data data.yaml --epochs 100 --batch-size 16 --img-size 640 --weights yolov8m.pt`

For inference on a new image:
`python detect.py --weights runs/detect/train44/weights/best.pt --source path_to_image`

### Future Improvements
- Explore other models like YOLOv8 or EfficientDet for improved precision and recall.
- Implement post-processing techniques to improve accuracy in detecting overlapping or occluded leaves.
