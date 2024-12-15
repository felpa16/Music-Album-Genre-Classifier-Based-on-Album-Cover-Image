from unittest.mock import DEFAULT
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from cnn_data_handling import train_loader, val_loader, mb_train_dataset
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report

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
        self.parameters = model.parameters()      
    
    # def train(self, train_loader, val_loader, loss_fn, optimizer, epochs=10):
        
    #     for i in range(0, epochs):
    #         self.model.train()
    #         train_loss = 0
    #         for x, y in tqdm(train_loader):
    #             x, y = x.to(self.device), y.to(self.device)
    #             optimizer.zero_grad()
    #             outputs = self.model(x)
    #             loss = loss_fn(outputs, y)
    #             loss.backward()
    #             optimizer.step()
    #             train_loss += loss.item()

    #         print(f"Epoch {i+1}, Training Loss: {train_loss / len(train_loader)}")
            
    #         self.model.eval()
    #         val_loss = 0
    #         with torch.no_grad():
    #             for x, y in val_loader:
    #                 x, y = x.to(self.device), y.to(self.device)
    #                 outputs = self.model(x)
    #                 loss = loss_fn(outputs, y)
    #                 val_loss += loss.item()

    #             print(f"Epoch {i+1}, Validation Loss: {val_loss / len(val_loader)}")

    def train(self, train_loader, val_loader, loss_fn, optimizer, epochs=10, patience=3):
        
        best_val_accuracy = 0.0
        early_stop_counter = 0

        for i in range(0, epochs):
            self.model.train()
            train_loss, correct_train = 0, 0
            for x, y in tqdm(train_loader):
                print(y)
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                correct_train += (outputs.argmax(1) == y).sum().item()
            
            train_accuracy = correct_train/len(train_loader.dataset)

            print(f"Epoch {i+1}, Training Loss: {train_loss / len(train_loader.dataset)}")
            print(f"Epoch {i+1}, Training Accuracy: {train_accuracy}")
            
            self.model.eval()
            val_loss, correct_val = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = loss_fn(outputs, y)
                    val_loss += loss.item()
                    correct_val += (outputs.argmax(1) == y).sum().item()

            val_accuracy = correct_val/len(train_loader.dataset)
            print(f"Epoch {i+1}, Validation Loss: {val_loss / len(val_loader)}")
            print(f"Epoch {i+1}, Validation Accuracy {val_accuracy}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"No improvement for {early_stop_counter} epoch(s)")
            
            if early_stop_counter == patience:
                print(f"Early stopping triggered after {early_stop_counter} epochs")
                break

    def train_with_mini_batch(self, train_data, val_loader, loss_fn, optimizer, batch_size=32, epochs=10, patience=3):

        best_val_accuracy = 0.0
        early_stop_counter = 0

        for epoch in tqdm(range(0, epochs)):
            train_loss, correct_train = 0, 0

            indices = np.random.choice(len(train_data), size=batch_size, replace=False)
            batch_samples = [train_data[i] for i in indices]
            x, y = zip(*batch_samples)
            x = torch.stack([torch.tensor(img, dtype=torch.float32) for img in x])
            y = torch.tensor(y, dtype=torch.long)

            self.model.train()
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct_train += (outputs.argmax(1) == y).sum().item()
            
            train_accuracy = correct_train/batch_size

            print(f"Epoch {epoch+1}, Training Loss: {train_loss / len(train_loader.dataset)}")
            print(f"Epoch {epoch+1}, Training Accuracy: {train_accuracy}")
                

            if epoch == 9:    
                self.model.eval()
                val_loss, correct_val = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        outputs = self.model(x)
                        loss = loss_fn(outputs, y)
                        val_loss += loss.item()
                        correct_val += (outputs.argmax(1) == y).sum().item()

                val_accuracy = correct_val/len(train_loader.dataset)
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}")
                print(f"Epoch {epoch+1}, Validation Accuracy {val_accuracy}")

                # if val_accuracy > best_val_accuracy:
                #     best_val_accuracy = val_accuracy
                #     early_stop_counter = 0
                # else:
                #     early_stop_counter += 1
                #     print(f"No improvement for {early_stop_counter} epoch(s)")
                
                # if early_stop_counter == patience:
                #     print(f"Early stopping triggered after {early_stop_counter} epochs")
                #     break

    
    def forward(self, image):
        input_tensor = transforms.ToTensor()(image).unsqueeze(0)
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        prediction = torch.argmax(output, dim=1).item()

        return prediction
    

    def get_parameters(self):
        return self.parameters

    def save(self):
        model_filename = f"resnet50_model"
        torch.save({'model_state_dict': self.model.state_dict()}, model_filename)

    def confusion_mat(self):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                all_preds.append(outputs.argmax(1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)


        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(all_labels, all_preds))


        conf_mat = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()



def prepare_cnn(outputs):
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for i in model.parameters():
        i.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, outputs),
        nn.Softmax(dim=1)
    )

    return model

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None or metric > self.best_score + self.delta:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CNN(outputs=15)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.get_parameters(), lr=0.001)
    
    # model.train(train_loader, val_loader, loss_fn, optimizer, epochs=10)
    model.train_with_mini_batch(mb_train_dataset, val_loader, loss_fn, optimizer, batch_size=32, epochs=10, patience=3)

    model.save()

    model.confusion_mat()

if __name__ == '__main__':
    main()