#data prep
import numpy as np 
import torch
import math
from torch.utils.data import Dataset
from typing import List
import pandas as pd
import torch.nn as nn
from PIL import Image
import os 

csvname = 'UTKFaceAugmented.csv'
dataset = csvname

df = pd.read_csv(dataset)

print(len(df))
print(df.columns)
print(df.info())
df.drop(columns=['gender', 'race', 'uses_skincare'])
# I dropped these columns since they have no effect age on paper

categoric_columns = ['has_tiktok' , 'remembers_disco','max_annual_earnings', 'num_haircuts_life']
for i in range(len(categoric_columns)):
    print("Column: {categoric_columns[i]}")
    counts = df[categoric_columns[i]].value_counts()
    for label, count in counts.items():
            print("Label: '{label}' | Frequency: {count}")
    
keep_categoric_columns = ['has_tiktok' , 'remembers_disco','max_annual_earnings', 'num_haircuts_life']
for col in keep_categoric_columns:
     df = df.join(pd.get_dummies(df[col], dtype = 'int', prefix=col+'_cat'), how = 'outer')
    #populate data as 1's and 0's"



# image data prep
image_dir = r'\Users\jqvil\Desktop\jessica quenneville\images\images'
csv_file = 'UTKFaceAugmented.csv'

data = pd.read_csv(csv_file)
# Placeholder to store image data and labels
images = []
labels = []


# Load and preprocess images
for index, row in data.iterrows():
    filename = row['filename']  
    age_label = row['age']  
    image_path = os.path.join(image_dir, filename)
    if os.path.exists(image_path):  # Check if the file exists before processing
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((64, 64))
        img_tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
        images.append(img_tensor.unsqueeze(0))  # Add a batch dimension
        labels.append(age_label)
    else:
        print(f"File not found: {image_path}")


image_tensor = torch.cat(images, dim=0)
labels = torch.tensor(labels, dtype=torch.float32)

