import torch
import torch.nn as nn

class MLPmodule(torch.nn.Module):
    """
    This is the 2-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(MLPmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.num_bottleneck = 512
        self.classifier = nn.Sequential(
                                       nn.ReLU(),
                                       nn.Linear(self.num_frames * self.img_feature_dim,
                                                 self.num_bottleneck),
                                       #nn.Dropout(0.90), # Add an extra DO if necess.
                                       nn.ReLU(),
                                       nn.Linear(self.num_bottleneck,self.num_class),
                                       )
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input


def return_MLP(relation_type, img_feature_dim, num_frames, num_class):
    MLPmodel = MLPmodule(img_feature_dim, num_frames, num_class)

    return MLPmodel
