import torch
#from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def evaluate(model, data_loader, device):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    return all_predictions, all_targets


#def compute_metrics(predictions, targets, num_classes=24):
    all_preds = []
    all_targets = []

    for pred, target in zip(predictions, targets):
        for label, score in zip(pred['labels'], pred['scores']):
            if score > 0.5:
                all_preds.append(label.item())
                all_targets.append(target['labels'][0].item())  # Assuming each image has one target label

    conf_matrix = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    acc = accuracy_score(all_targets, all_preds)

    return acc, conf_matrix
