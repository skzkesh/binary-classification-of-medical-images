# Binary Classification of Medical Images using ResNet-18
![thyroid_test prediction](test_predictions.png)
![trachea_test_prediction](trachea_test_predictions.png)

This project uses ResNet-18 to classify ultrasound images according to the presence of a specific feature.

## Binary Classification
Binary classification refers to the process of analyzing data to categorize items into one of two distinct classes. A binary classification model is employed when the goal is to classify an item into one of two categories. This project focuses on applying binary classification to medical images, specifically classifying them as either "organ absent" or "organ present." We successfully trained and implemented this binary classification model to detect the presence of the thyroid and trachea in a series of ultrasound images with high precision.

## ResNet-18
In this project, we use ResNet-18 as the pre-trained model. This choice is due to the limited size of our dataset, which was collected independently and has less variance. ResNet-18 performs well with smaller datasets and requires less computational power compared to models like ResNet-50 or ResNet-101, making it faster and more efficient. A simpler model like ResNet-18 reduces the risk of overfitting while still capturing unique medical patterns. Transfer learning allows us to apply pre-trained models to different types of images, including medical images such as ultrasound scans.

## Pre-requisite
- Windows/Linux/MacOS
- Python
- CPU or GPU

## Organize folder directory 
In PyTorch, the most common way of doing image classification is to structure your dataset into folders by class. The `torchvision.datasets.ImageFolder` utility takes folder names as class labels and automatically assigns labels to each item, eliminating the need for manual labeling.

First, create two main folders named "train" and "validation" in your working directory. The "train" subfolder will store images used for training, while the "validation" folder will hold validation images. The typical split is 80/20. Within each of these folders, create two subfolders named "x_present" and "x_absent," where "x" is replaced by the organ you are trying to classify. Load your data according to these labels (folder names).

## Setup
Before running the code, ensure you have downloaded all the required libraries. Use the following commands to install the necessary packages:
- `pip install torch`
- `pip install torchvision`
- `pip install scikit-learn`

## Running the code
Once all libraries are installed, execute your code via the terminal or using the run button in your IDE. After successful execution, details about the current epoch, loss, and validation accuracy will be displayed. If the current model performs better than the previous one, it will be saved in a `.pth` format. This file can later be used to load weights for testing and real-world applications.

## Optimization
Your model's performance can be improved over time through hyperparameter tuning. This can be accomplished using various methods, including GridSearchCV, RandomizedSearchCV, and Bayesian Optimization. Additionally, we aim to collect a larger dataset to enhance the model's generalization and accuracy.

## Notes
This is the google drive of .pth file for thyroid and trachea classification. [link](https://drive.google.com/drive/folders/1_X0oAMHWVAe2Icy68qbbT1MGAnDhgtVT?usp=sharing)


