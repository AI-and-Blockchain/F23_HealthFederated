import torch
import torch.nn as nn
import torchvision

class ClientModel2Layers(nn.Module):
    def __init__(self):
        super(ClientModel2Layers, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(27136, 64),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.resnet(x)
        x2 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)

        # Final embedding has size 64 and in range [-1, 1]
        x = self.classifier(x)

        return x
    
class ClientModel3Layers(nn.Module):
    def __init__(self):
        super(ClientModel3Layers, self).__init__()
        self.densenet = torchvision.models.densenet169(pretrained=True)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.vgg = torchvision.models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(108672, 64),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.densenet(x)
        x2 = self.resnet(x)
        x3 = self.vgg(x)

        # Flatten and concatenate
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        # Final embedding has size 64 and in range [-1, 1]
        x = self.classifier(x)

        return x

class ServerModel(nn.Module):
    def __init__(self):
        super(ServerModel, self).__init__()

        # classification layers
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.BatchNorm1d(64),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        return x
