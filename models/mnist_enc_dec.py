import torch.nn as nn

class MnistEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MnistEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc3 = nn.Linear(400, output_dim)
        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 28 * 28)
        h = self.act(self.fc1(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z

class MnistDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MnistDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 500),
            nn.Tanh(),
            nn.Linear(500, 784)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 28, 28)
        return mu_img