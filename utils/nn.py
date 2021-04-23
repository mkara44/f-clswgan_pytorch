import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, out_dim, x_dim=2048):
        super().__init__()

        self.fc1 = nn.Linear(x_dim, int(x_dim / 2))
        self.fc2 = nn.Linear(int(x_dim / 2), int(x_dim / 2))
        self.fc3 = nn.Linear(int(x_dim / 2), int(x_dim / 2))
        self.fc4 = nn.Linear(int(x_dim / 2), out_dim)

    def forward(self, x):
        for fc in [self.fc1, self.fc2, self.fc3]:
            x = F.leaky_relu(fc(x))
            x = F.dropout(x)

        x = self.fc4(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, attr_dim):
        super().__init__()

        self.fc1 = nn.Linear(attr_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)

        self.apply(weights_init)

    def forward(self, noise, atts):
        x = torch.cat((noise, atts), 1)
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, attr_dim, x_dim=2048):
        super().__init__()

        self.fc1 = nn.Linear(x_dim + attr_dim, 4096)
        self.fc2 = nn.Linear(4096, 1)

        self.apply(weights_init)

    def forward(self, feat, atts):
        x = torch.cat((feat, atts), 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    from config import cfg

    nn = Generator(cfg.attr_number + cfg.latent_dim)
    print(nn)
