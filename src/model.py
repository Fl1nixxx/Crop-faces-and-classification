import torch
import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer3.parameters():
            param.requires_grad = True

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class FilterNet(nn.Module):
    def __init__(self, num_age_classes = 9, num_race_classes = 7, num_gender_classes = 1):
        super().__init__()

        self.backbone = Backbone()

        self.age_classifier = nn.Sequential(

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(1024, num_age_classes - 1),)

        self.race_classifier = nn.Sequential(

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, num_race_classes),)

        self.gender_classifier = nn.Sequential(

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(512, num_gender_classes),)

    def forward(self, x):

        features = self.backbone(x)

        age_logits = self.age_classifier(features)
        race_logits = self.race_classifier(features)
        gender_logits = self.gender_classifier(features)

        return age_logits, race_logits, gender_logits

def load_model(weights_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FilterNet()

    state_dict = torch.load(weights_path,map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model
