from torchvision.models import resnet50
from torch import nn

def prepare_cnn(outputs):
    
    model = resnet50(pretrained=True)

    for i in model.parameters():
        i.requires_grad = False
    
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, outputs),
        nn.Softmax(dim=1)
    )

    return model

