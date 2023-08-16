import torch
import torch.nn as nn

class Cifar10Classifier(torch.nn.Module):

    def __init__(self):
        super(Cifar10Classifier, self).__init__()

        self.features = torch.nn.Sequential(                                                        # inp: B 3 32 32
            torch.nn.BatchNorm2d(num_features=3),
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),    # B 64 32 32
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),   # B 64 32 32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                 # B 64 16 16
            torch.nn.Dropout2d(0.2),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # B 128 16 16
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # B 128 16 16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                 # B 128 8 8
            torch.nn.Dropout2d(0.2),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # B 256 8 8
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # B 256 8 8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                                 # B 256 4 4
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            torch.nn.Dropout2d(0,2),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0),
            torch.nn.Dropout2d(0,2),
            torch.nn.BatchNorm2d(num_features=512),
        )

        self.decider = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2 * 2 * 512, 128),                                                      # B 4 * 4 * 256
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)                                                                # B 10
        )

    def forward(self,x):
        features = self.features(x)
        features = nn.functional.adaptive_avg_pool2d(features,(2,2))
        features = features.flatten(1)
        output = self.decider(features)
        return output
