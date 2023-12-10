#networks.py
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import tqdm


input_size = 16
#sample random data for input
class CNNClassifier(nn.Module):
    def __init__(self, output_dim: int):
        super(CNNClassifier, self).__init__()
        assert output_dim > 0, "Output dimension must be a positive integer"
        self.conv1 = nn.Conv2d(
            in_channels = 200,
            out_channels = 16,
            kernel_size = (3, 1), 
            stride = (1, 1),
            padding = (1, 1)
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (1,1)
        )
        self.conv2 = nn.Conv2d(
            in_channels = 16, 
            out_channels = 64, 
            kernel_size = (3, 3), 
            stride = (1, 1), 
            padding = (0, 0)
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (1,1)
        )
        self.linear1 = nn.Linear(
            in_features=64,
            out_features=output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool2(x)
        # reshape for linear layer
        # note that the output of maxpool 2 is (*,64,1,1) so we just need to take the first column and row. 
        # If the output size is not 1,1, we have to flatten x before going into linear using torch.flatten
        x = x[:,:,0,0] 
        x = self.linear1(x)     
        x = torch.sigmoid(x)  
        return x

# Define multimodal neural network
class MultiModalNet(nn.Module):
    def __init__(self, image_model, num_tabular_features):
        super(MultiModalNet, self).__init__()
        self.image_model = image_model
        self.fc_tabular = nn.Linear(num_tabular_features, 2176) 
        self.fc_combined = nn.Linear(2176 + 2048, 1)  # Combine image and tabular features

    def forward(self, image):
        image_features = self.image_model(image)
        image_features = image_features.view(image_features.size(0), -1)
        tabular_features = torch.relu(self.fc_tabular())
        combined = torch.cat((image_features, tabular_features), dim=1)
        output = self.fc_combined(combined)
        return output



import torch
import math
from PIL import Image

class CustomMultimodalDataloader():
    """
    Wraps a multimodal dataset and enables fetching one batch at a time
    """
    def __init__(self, image_paths: list, captions: list, labels: list, batch_size: int = 1, randomize: bool = False):
        self.image_paths = image_paths
        self.captions = captions
        self.labels = labels
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)
        self.fc1 = nn.Linear(16 * 20 * 20, 128)
        self.fc2 = nn.Linear(128, 10)  
        
    def get_length(self):
        return len(self.image_paths)

    def randomize_dataset(self):
        """
        This function randomizes the dataset while maintaining the relationships between
        image paths, captions, and labels.
        """
        indices = torch.randperm(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.captions = [self.captions[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time.
        """
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            start_idx = b_idx * self.batch_size
            end_idx = min((b_idx + 1) * self.batch_size, len(self.image_paths))
            image_batch = [
                self.load_image(self.image_paths[i]) for i in range(start_idx, end_idx)
            ]
            caption_batch = self.captions[start_idx:end_idx]
            label_batch = self.labels[start_idx:end_idx]

            batches.append({
                'image_batch': image_batch,
                'caption_batch': caption_batch,
                'label_batch': label_batch,
                'batch_idx': b_idx,
            })

        self.iter = iter(batches)

    def load_image(self, image_path):
        """
        Function to load an image from disk given the image path.
        Modify this function according to your image loading and preprocessing needs.
        """
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((20, 20))        
        return image

    def fetch_batch(self):
        """
        This function retrieves the next batch and handles the end of an epoch
        by regenerating the iterator.
        """
        if self.iter is None:
            self.generate_iter()

        try:
            batch = next(self.iter)
        except StopIteration:
            self.generate_iter()
            batch = next(self.iter)

        return batch
