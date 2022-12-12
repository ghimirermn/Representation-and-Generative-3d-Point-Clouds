import torch
import torch.nn as nn

class LatentGenerator(nn.Module):
    """
    Attributes:
        input shape: [batch_size, 128]
        output shape: [batch_size, 2048,3]
    """

    def __init__(self):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(in_features=128, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=6144, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=6144, out_features=6144,bias=True),
        )

    def forward(self, x):
        g1 = self.generator(x)
        return (g1.reshape(-1, 2048,3))



class LatentDiscriminator(nn.Module):
    """
    Attributes:
        input shape: [batch_size, 2048, 3]
        output shape: [batch_size, 1]
    """

    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=6144, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1, out_features=1, bias=True)
        )
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        flatten = self.flatten(x)
        d1 = self.discriminator(flatten)
        d1_sig = self.sigmoid(d1)
        return (d1_sig)

if __name__ == "__main__":
    input = torch.randn(4,2048,3).to('cuda')
    # model = LatentGenerator().to('cuda')
    model = LatentDiscriminator().to('cuda')
    output = model(input)
    print(output.shape)