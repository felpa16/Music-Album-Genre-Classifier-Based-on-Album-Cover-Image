from unittest.mock import DEFAULT
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from cnn_data_handling import train_loader, val_loader
from tqdm import tqdm
from torchvision import transforms

class CNN:
    def __init__(self, outputs):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        for i in model.parameters():
            i.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, outputs)
        )
        self.model = model

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model.to(self.device)

        
    
    def train(self, train_loader, val_loader, loss_fn, optimizer, epochs=10):
        
        for i in range(0, epochs):
            self.model.train()
            train_loss = 0
            for x, y in tqdm(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                print(f"Epoch {i+1}, Training Loss: {train_loss / len(train_loader)}")
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = loss_fn(outputs, y)
                    val_loss += loss.item()

                print(f"Epoch {i+1}, Validation Loss: {val_loss / len(val_loader)}")
    
    def forward(self, image):
        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        prediction = torch.argmax(output, dim=1).item()

        return prediction


def prepare_cnn(outputs):
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for i in model.parameters():
        i.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, outputs),
        nn.Softmax(dim=1)
    )

    return model

def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=10):
    
    for i in range(0, epochs):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            print(f"Epoch {i+1}, Training Loss: {train_loss / len(train_loader)}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()

            print(f"Epoch {i+1}, Validation Loss: {val_loss / len(val_loader)}")


def main():
    
    model = prepare_cnn(outputs=16)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    train(model, train_loader, val_loader, loss_fn, optimizer, epochs=20)

if __name__ == '__main__':
    main()