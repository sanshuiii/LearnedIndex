import BasicFS
import NN
import torch

if __name__=="__main__":
    bs = BasicFS.DataProvider('sorted_demo_data')
    x, y = bs.gen_test_data()

    mxx = max(x)
    mxy = max(y)
    mix = min(x)
    miy = min(y)
    for i in range(len(x)):
        x[i] = (x[i]-mix)/(mxx-mix)
        y[i] = (y[i]-miy)/(mxy-miy)

    x = torch.tensor(x, requires_grad=False).view(-1,1)
    y = torch.tensor(y, requires_grad=False).view(-1,1)

    net = NN.NeuralNet()

    print(x)

