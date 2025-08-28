import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
 
# Store device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
 
# Load test dataset
test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model definition
model = models.resnet18(pretrained=False)  # Initialize architecture
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
model.load_state_dict(torch.load('best_model.pth', map_location=device))  # Load existing model
model = model.to(device)
model.eval()

# Inference loop
y_true = []
y_pred = []
 
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # shape (batch, 1)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)  # convert logits to binary
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
 
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Make predictions
import matplotlib.pyplot as plt
import torchvision
class_names = test_dataset.classes
 
# Get one batch
images, labels = next(iter(test_loader))
images = images.to(device)
outputs = model(images)
preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
 
# Show images with predictions
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i, ax in enumerate(axes.flatten()):
    img = images[i].cpu().permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # unnormalize
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"P: {class_names[int(preds[i][0])]}\nT: {class_names[labels[i].item()]}")
    ax.axis('off')

plt.show()
plt.savefig('test_predictions.png')

