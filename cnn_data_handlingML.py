import numpy as np
import torch
from torch.utils.data import Sampler, Dataset

train_data_path = "/Users/felipe/ML/Proyecto_Final/Proyecto_Final_ML/classified_audios_train.csv"
test_data_path = "/Users/felipe/ML/Proyecto_Final/Proyecto_Final_ML/classified_audios_test.csv"
valid_data_path = "/Users/felipe/ML/Proyecto_Final/Proyecto_Final_ML/classified_audios_valid.csv"

class MyDataset(Dataset):
    def __init__(self, path):
        self.dataset, self.labels = read_cnn_data(path)

    def __getitem__(self, index):
        x = self.dataset[index]
        y = self.labels[index]
        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.dataset)

class MiniBatchSampler(Sampler):
    def __init__(self, dataset, size):
        self.size = size
        self.dataset = dataset

    def __iter__(self):
        samples = np.random.choice(len(self.dataset), self.size, replace=False)
        return iter(samples)
    
    def __len__(self):
        return self.size

def read_cnn_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    dataset = []
    labels = []
    for i in range(1, len(lines)):
        sample = lines[i].split(",")
        dataset.append([float(sample[i]) for i in range(0, len(sample) - 2)])
        labels.append(int(sample[-2]) - 1)
    
    return dataset, labels