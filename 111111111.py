import torch
import torch.nn as nn
import torch.nn.functional as F
from test import prepare_dataloader, train, test
import torch.optim as opt
from resnet import *

class myconv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_qat):
        super(myconv, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.is_qat = is_qat
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        return super(myconv, self).forward(x)



class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = myconv(in_channels=3, out_channels=16, kernel_size=5, stride=1, is_qat=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1296, 128)
        self.fc2 = nn.Linear(128, 10)
        # self.quant=torch.quantization.QuantStub()
        # self.dequant=torch.quantization.DeQuantStub()
        # self.fc3 = nn.Linear(36,10)

    def forward(self, x):
        # qx=self.quant(x)
        if self.conv1.is_qat:
            x=self.conv1.quant(x)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.conv1.dequant(x)
        else:
            x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))
        # x = self.aap(x)
        # x = x.view(x.shape[0],-1)
        # x = self.fc3(x)
        x = x.contiguous().view(x.size()[0], -1)
        # print("x.shape:{}".format(x.shape))
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # x=self.dequant(x)
        return x


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 64
    test_batch_size = 64
    lr = 0.005
    momentum = 0.8
    epochs = 30

    torch.manual_seed(1)
    train_loader, test_loader = prepare_dataloader(train_batch_size, test_batch_size)

    model = ResNet18()
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # model.fuse_model()
    prepared_model = torch.quantization.prepare_qat(model, inplace=False)
    optimizer = opt.SGD(prepared_model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, 2):
        train(prepared_model.to(device), device, train_loader, optimizer, epoch)
        print('finish training!!!!!!\n')
        prepared_model.to('cpu')
        convert_model = torch.quantization.convert(prepared_model.eval(), inplace=False)
        test(convert_model.to('cpu'), device='cpu', test_loader=test_loader)