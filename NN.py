import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, input):
        hidden = self.fc1(input)
        output = self.fc2(hidden)
        return output


if __name__ == "__main__":
    net = NeuralNet()
    b = torch.tensor([1.0, 0.0])
    print(net(b))

    optim = torch.optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(10000):
        loss = criterion(net(b), torch.tensor(3.0))
        optim.zero_grad()
        loss.backward()
        optim.step()

    b = net(torch.tensor([1.0, 0.0]))
    print(b)
