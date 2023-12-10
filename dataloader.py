import torch
import math
import torch.nn as nn
from PIL import Image
class CustomDataloader():

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):

        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def generate_iter(self):

    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):

        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch




class CustomImageDataloader():
    """
    Wraps a dataset and enables fetching of one batch at a time
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, randomize: bool = False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return self.x.shape[0]

    def randomize_dataset(self):
        """
        This function randomizes the dataset, while maintaining the relationship between 
        x and y tensors
        """
        indices = torch.randperm(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]
    def resize_images(self, desired_size):
        resized_images = []
        for img in self.x:
            resized_img = torch.nn.functional.interpolate(img.unsqueeze(0), size=desired_size, mode='bilinear', align_corners=False)
            resized_images.append(resized_img.squeeze(0))
        self.x = torch.stack(resized_images)
    def generate_iter(self):
        """
        This function converts the dataset into a sequence of batches, and wraps it in
        an iterable that can be called to efficiently fetch one batch at a time
        """
    
        if self.randomize:
            self.randomize_dataset()

        # split dataset into sequence of batches 
        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            batches.append(
                {
                'x_batch':self.x[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'y_batch':self.y[b_idx * self.batch_size : (b_idx+1) * self.batch_size],
                'batch_idx':b_idx,
                }
            )
        self.iter = iter(batches)
    
    def fetch_batch(self):
        """
        This function calls next on the batch iterator, and also detects when the final batch
        has been run, so that the iterator can be re-generated for the next epoch
        """
        # if the iter hasn't been generated yet
        if self.iter == None:
            self.generate_iter()

        # fetch the next batch
        batch = next(self.iter)

        # detect if this is the final batch to avoid StopIteration error
        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            # generate a fresh iter
            self.generate_iter()
        return batch

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
        self.fc2 = nn.Linear(128, 10)  # Assuming num_classes is the number of output classes

        if self.randomize:
            self.randomize_dataset()

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
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = image.resize((20, 20))
            return image
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
            # Return a default value or handle the exception accordingly
            return None 


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
    def __iter__(self):
        for b_idx in range(self.num_batches_per_epoch):
            start_idx = b_idx * self.batch_size
            end_idx = min((b_idx + 1) * self.batch_size, len(self.image_paths))
            image_batch = [
                self.load_image(self.image_paths[i]) for i in range(start_idx, end_idx)
            ]
            caption_batch = self.captions[start_idx:end_idx]
            label_batch = self.labels[start_idx:end_idx]

            yield {
                'image_batch': image_batch,
                'caption_batch': caption_batch,
                'label_batch': label_batch,
                'batch_idx': b_idx,
            }