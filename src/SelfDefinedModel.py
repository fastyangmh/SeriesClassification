#import
import torch
import torch.nn as nn


#class
class SelfDefinedModel(nn.Module):
    def __init__(self, in_features, num_classes) -> None:
        super().__init__()
        layers = []
        for idx in range(4):
            out_features = in_features * 4
            layers.append(
                nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.GLU())
            layers.append(nn.Dropout(p=0.25))
            in_features = out_features // 2
        layers.append(
            nn.Linear(in_features=in_features, out_features=num_classes))
        self.model = nn.Sequential(*layers)

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
