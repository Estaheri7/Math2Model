# Brain Tumor Detection

This project focuses on brain tumor segmentation using deep learning techniques. The primary goal is to accurately segment brain tumors from MRI images.

## Usage

To run the project, follow these steps:

1. Prepare the dataset as described in the [Dataset](#dataset) section.
2. Train the model using the provided Jupyter notebook or Python scripts.
3. Evaluate the model on the test dataset.

## Dataset

The dataset used in this project consists of MRI images with corresponding annotations for brain tumors. The annotations are in COCO format. This dataset can be downloaded from Kaggle [here](https://www.kaggle.com/).

### Directory Structure

```
data/
├── brain/
│   ├── train/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   ├── test/
│   │   ├── images/
│   │   └── _annotations.coco.json
│   └── valid/
│       ├── images/
│       └── _annotations.coco.json
```

## Model Architecture

The project implements a custom U-Net architecture for brain tumor segmentation. Additionally, transfer learning is applied using a pre-trained ResNet34 model with an attention mechanism.

## Training

To train the model, use the provided Jupyter notebook `BRAIN_TUMOR_SEGMENTATION.ipynb`. The notebook includes the following steps:

1. Importing libraries
2. Setting up the device
3. Defining custom dataset and dataloaders
4. Implementing the U-Net model
5. Defining loss functions and metrics
6. Training the model
7. Evaluating the model

## Evaluation

The model is evaluated using the Dice coefficient, which measures the overlap between the predicted and ground truth masks.

## Results

The training and validation results, including loss and Dice coefficient, are plotted for analysis. The model's performance on the test dataset is also displayed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.