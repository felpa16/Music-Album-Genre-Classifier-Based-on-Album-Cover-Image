import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader

class MyDataset(Dataset):
    def __init__(self, dataframe, path):
        self.dataframe = dataframe
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_name = os.path.join(self.path, self.dataframe.iloc[index, 0])
        image = Image.open(img_name).convert("RGB")
        image = self.transform(image)
        label = self.dataframe.iloc[index, 1]

        return image, label

    def __len__(self):
        return len(self.dataframe)

labels_df = pd.read_csv('labels.csv')

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['label'])
val_df, test_df = train_test_split(val_df, test_size=0.5, stratify=val_df['label'])

train_data = MyDataset(train_df, "/Users/felipe/Vision/tp_final/album_covers")
val_data = MyDataset(val_df, "/Users/felipe/Vision/tp_final/album_covers")
test_data = MyDataset(test_df, "/Users/felipe/Vision/tp_final/album_covers")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)