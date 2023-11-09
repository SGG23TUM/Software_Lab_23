import torch
from data_loader import get_data_loaders
from faster_rcnn_model import get_faster_rcnn_model
from train import train_one_epoch
from test import evaluate
import numpy as np


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
batch_size = 8
num_classes = 23 + 1  # Background + 23 classes (ceiling, lighting, speaker ...)
num_epochs = 2
lr = 0.02

# Get data loaders
train_loader, test_loader = get_data_loaders(batch_size=batch_size)

# Initialize model and optimizer
model = get_faster_rcnn_model(num_classes=num_classes)
model.load_state_dict(torch.load('src/faster_rcnn_model.pth'))
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

# Training loop
for epoch in range(num_epochs):
    avg_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "src/faster_rcnn_model.pth")

# Test
predictions, targets = evaluate(model, test_loader, device)
# Compute metrics and results (to be added)


import matplotlib.pyplot as plt
import torchvision.transforms as T

def plot_predictions(images, predictions):
    fig, ax = plt.subplots(1, len(images), figsize=(40, 20))

    for img, pred, a in zip(images, predictions, ax):
        img = T.ToPILImage()(img.cpu()).convert("RGB")
        boxes = pred['boxes'].cpu().detach().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().detach().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score > 0.5:  # Display boxes with scores greater than 0.5
                random_color = np.random.rand(3,)
                x_min, y_min, x_max, y_max = box
                a.imshow(img)
                a.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=random_color, linewidth=1))
                a.text(x_min, y_min, f"{label}: {score:.2f}", color=random_color)

    plt.show()

# After testing
sample_images, _ = next(iter(test_loader))
sample_predictions = model(sample_images)
plot_predictions(sample_images, sample_predictions)


# Compute Accuracy and Confusion Matrix
#accuracy, conf_mat = compute_metrics(predictions, targets)
#print(f"Accuracy: {accuracy * 100:.2f}%")

# Display Confusion Matrix
#import seaborn as sns
#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 7))
#sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
#plt.xlabel('Predicted labels')
#plt.ylabel('True labels')
#plt.title('Confusion Matrix')
#plt.show()