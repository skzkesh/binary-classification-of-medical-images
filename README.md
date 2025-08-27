# binary-classification-of-medical-images
Binary Classification is defined as the process of analyzing data and making decision between two elements. Binary classification model is used when we want to classify an item to one of two categories. This project focuses on applying binary classification to classifiy medical images into two categories, such as "organ absent" and "organ present". Previously, we successfully implemented this model to detect the present of thyroid and trachea in a sequence of ultrasound images with high accuracy.

## Organize folder directory 
In PyTorch, the most common way of doing image classification is to structure your dataset into folders by class. `torchvision.datasets.ImageFolder` will take folder name as class and it automatically assigns labels to each item in the folder according to the class, removing the need for manual labelling. Firstly, we create two folders named "train" and "validation". Train folder is used to store images for training while validation folder store validation images. The most common general-purpose split is 80 / 20. After creating both folders, create two subfolders in each folder, each named "x present" and "x absent" where x can be substitute with the organ that you are trying to classify.

# Pre-trained model
This project use ResNet-18 pre-trained model for binary classification. The main reason is because the size of data which was small. ResNet requires lighter computational power compared to ResNet=50-101 which makes it faster and more efficinnt. A smaller model like ResNet-18 reduced te risk overfitting, while still able to capture complex medical pattern. Medical images such as ultrasound images differ from natural images. However, transfer learning enable pre-trained model to be implemented in different type of images.



