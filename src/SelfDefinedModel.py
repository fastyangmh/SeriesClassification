#import
import torch
import torch.nn as nn


#class
class SelfDefinedModel(nn.Module):
    def __init__(self, in_features, num_classes) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features * 4),
            nn.BatchNorm1d(num_features=in_features * 4), nn.GLU(),
            nn.Linear(in_features=in_features * 2, out_features=num_classes))

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    #parameters
    in_features = 30
    num_classes = 2
    batch_size = 32

    #create model
    model = SelfDefinedModel(in_features=in_features, num_classes=num_classes)

    #create input data
    x = torch.rand(batch_size, in_features)

    #get model output
    y = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y.shape))
