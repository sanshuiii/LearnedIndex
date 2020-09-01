import torch
from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(1, 2000)
        self.fc2 = nn.Linear(2000, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, input):
        hidden = nn.functional.relu(self.fc1(input))
        hidden = nn.functional.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        return output

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(1,1)

    def forward(self, input):
        output = self.fc(input)
        return output


def run_session(model, input, output, is_train=False):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    if is_train:
        model.train()
        predict = model(input)
        loss = criterion(output, predict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
    else:
        model.eval()
        with torch.no_grad():
            predict = model(input)
            loss = criterion(output, predict)
            return loss, predict


if __name__ == "__main__":
    net = NeuralNet()
    b = torch.tensor([0.3])
    print(net(b))

    for i in range(1000):
        loss = run_session(net, b, torch.tensor([0.7]), True)
        if i % 100 == 0:
            print(loss)

    b = net(torch.tensor([0.3]))
    print(b)
