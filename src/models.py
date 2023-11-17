import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet169(nn.Module):
    def __init__(self):
        super(DenseNet169, self).__init__()
        
        # Load pre-trained DenseNet169 model
        self.densenet = models.densenet169(pretrained=True)
        
        # Remove original input layer and classifier
        self.densenet.features = nn.Identity()
        self.densenet.classifier = nn.Identity()
        
        # Define a custom input layer to match the desired input shape
        self.input_layer = nn.Linear(64, 1664)  # 1664 is the output size of the original DenseNet169 features
        
        # Additional layers for downstream classification
        self.fc = nn.Linear(1664, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input_layer(x)
        features = self.densenet(x)
        output = self.fc(features)
        output = self.softmax(output)
        return output

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove original input layer and classifier
        self.resnet.conv1 = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # Define a custom input layer to match the desired input shape
        self.input_layer = nn.Linear(64, 2048)  # 2048 is the output size of the original ResNet50 features
        
        # Additional layers for downstream classification
        self.fc = nn.Linear(2048, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input_layer(x)
        features = self.resnet(x)
        output = self.fc(features)
        output = self.softmax(output)
        return output

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        # Load pre-trained VGG19 model
        self.vgg = models.vgg19(pretrained=True)
        
        # Remove original input layer and classifier
        self.vgg.features[0] = nn.Identity()  # Replace the original input layer
        self.vgg.classifier = nn.Identity()
        
        # Define a custom input layer to match the desired input shape
        self.input_layer = nn.Linear(64, 25088)  # 25088 is the output size of the original VGG19 features
        
        # Additional layers for downstream classification
        self.fc = nn.Linear(25088, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.input_layer(x)
        features = self.vgg(x)
        output = self.fc(features)
        output = self.softmax(output)
        return output

class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.densenet = models.densenet169(pretrained=True)
        self.resnet = models.resnet50(pretrained=True)
        self.vgg = models.vgg19(pretrained=True)

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(53760, 64),
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
        self.densenet = DenseNet169()
        self.resnet = ResNet50()
        self.vgg = VGG19()

        # Remove the classification layers (fully connected layers)
        self.densenet = nn.Sequential(*list(self.densenet.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        # Define new classification layers
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.BatchNorm1d(64), # Output features from densenet + resnet + vgg
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
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

        x = self.fc(x)
        return x

