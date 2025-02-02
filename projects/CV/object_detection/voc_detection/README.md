# Object Detection using Faster R-CNN on VOC Dataset

This project demonstrates object detection using the Faster R-CNN model trained on the Pascal VOC dataset.

## Setup


**Download the VOC dataset:**
   Ensure the VOC dataset is available in the `../../../data/voc_detection` directory. If not, set `download=True` in the dataset loading cell.

## Usage

1. **Import Libraries:**
   The necessary libraries are imported at the beginning of the notebook.

2. **Load and Split Dataset:**
   The VOC dataset is loaded and split into training, validation, and test sets.

3. **Custom Dataset Class:**
   A custom dataset class `ExtractVOCDataset` is defined to extract and process annotations from the VOC dataset.

4. **Data Transformation:**
   Images are resized and transformed into tensors.

5. **DataLoader:**
   Custom collate function is used to ensure each batch has the same size.

6. **Model Creation:**
   The Faster R-CNN model with a ResNet-50 backbone is created and modified for 21 classes.

7. **Training:**
   The model is trained for a specified number of epochs, and the training loss is printed after each epoch.

8. **Model Visualization:**
   Functions are provided to visualize the ground truth and predictions of the trained model.

9. **Evaluation:**
   The model's performance is evaluated using the Mean Average Precision (mAP) metric.

## Visualizing Predictions

To visualize the predictions of the trained model, use the `visualize_model` function. This function randomly selects an image from the test dataset and displays the ground truth and predicted bounding boxes.

## Evaluation Metrics

The evaluation metrics include mAP for different IoU thresholds and object sizes, as well as recall metrics.

## Saving the Model

The trained model is saved to a file named `model_voc_detection.pth`.
