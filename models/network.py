import torch
import torch.nn as nn
from torchsummary import summary


class PointAutoEncoder(nn.Module):
    """
    "Learning Representations and Generative Models for 3D Point Clouds"
    http://proceedings.mlr.press/v80/achlioptas18a/achlioptas18a.pdf

    Attributes:
            encoding_dim:2048
    """

    def __init__(self, encoding_dim=2048):
        super().__init__()

        self.encoding_dim = encoding_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(encoding_dim, 64, kernel_size=(1,), stride=(1,)),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(1,), stride=(1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.third_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=(1,), stride=(1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.fourth_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(1,), stride=(1,)),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.fifth_conv = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=(1,), stride=(1,)),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        self.decoding = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=6144, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=6144, out_features=6144,bias=True)
        )

    def forward(self, x):

        #encoding side
        first = self.first_conv(x)
        second = self.second_conv(first)
        third = self.third_conv(second)
        fourth = self.fourth_conv(third)
        fifth = self.fifth_conv(fourth)
        fifth_max = torch.max(fifth, dim=2, keepdim=False)[0]

        #decoding side
        decoder = self.decoding(fifth_max)
        return decoder

if __name__ == "__main__":
    model = PointAutoEncoder(encoding_dim=2048).to("cuda")
    input = torch.randn(1,2048,3).to("cuda")
    output = model(input)
    print(output.shape)
    summary(model, (2048,3))

