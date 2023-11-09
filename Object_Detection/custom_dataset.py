import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        super(CustomDataset, self).__init__()
        self.transforms = transforms
        self.annotation_dir = annotation_dir
        self.image_dir = image_dir

        self.image_files = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith('.png') or file.lower().endswith('.jpg')])  
        self.annotation_files = sorted([os.path.join(annotation_dir, file) for file in os.listdir(annotation_dir) if file.endswith('.json')])
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        annotation_path = self.annotation_files[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotations
        with open(annotation_path, 'r') as file:
            data = json.load(file)

        # Extract bounding boxes and labels
        boxes = []
        labels = []
        for obj in data["objects"]:
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2
            boxes.append([x_min, y_min, x_max, y_max])
            # Assuming first name in the 'names' list is the primary label
            labels.append(obj["names"][0])

        # Convert boxes into tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Convert labels (strings) to unique integers

        label_map = {
            "ceiling": 1,
            "lighting": 2,
            "speaker": 3,
            "wall": 4,
            "door": 5,
            "smoke alarm": 6,
            "floor": 7,
            "trash bin": 8,
            "elevator button": 9,
            "escape sign": 10,
            "board": 11,
            "fire extinguisher": 12,
            "door sign": 13,
            "light switch": 14,
            "emergency switch button": 15,
            "elevator": 16,
            "handrail": 17,
            "show window": 18,
            "pipes": 19,
            "staircase": 20,
            "window": 21,
            "radiator": 22,
            "stecker": 23
        }

        # Expand this map as needed
        labels = torch.tensor([label_map[label] for label in labels], dtype=torch.int64)

        # Construct target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)

        return image, target

if __name__ == "__main__":
    # Testing the Dataset
    dataset = CustomDataset(image_dir="data/images", annotation_dir="data/annotations")
    print(dataset[0])  # Print first item (image and target)
