import torch.nn as nn
import torchvision
from torchvision import models

class ConvLstm(nn.Module):
    def __init__(self, pretrain_nn,classifier_size):
        super(ConvLstm, self).__init__()
        conv_model = models.resnet152(pretrained=True)
        for param in conv_model.parameters():  #Todo find better way to do this
            param.requires_grad = False
        # changeing the clasiffer to a fully connected layer in the size we need.
        # right now all of the layers are unfreeze
        conv_model.fc = nn.Linear(conv_model.fc.in_features, classifier_size)
        self.conv_model = conv_model

    def forward(self,X):
        return self.conv_model(X)

