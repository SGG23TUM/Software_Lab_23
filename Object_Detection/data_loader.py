import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset

def get_data_loaders(batch_size):
    transform = T.Compose([
        T.ToTensor()
    ])
    
    dataset = CustomDataset(image_dir="src/Data/images", annotation_dir="src/Data/annotations", transforms=transform)

    # Define split ratios
    #train_size = int(0.8 * len(dataset))
    #test_size = len(dataset) - train_size

    # Modify this part in the data_loader.py
    train_size = int(0.80 * len(dataset))  # Use 20% of the dataset instead of 80%
    test_size = int(0.20 * len(dataset))  # Use 5% for testing
    remaining = len(dataset) - train_size - test_size  # Remaining data that won't be used
    train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, remaining])

    # Randomly split dataset
    #train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, test_loader




