import os
import sys
import time
import numpy
import torch
import torchvision
from torch import nn
from torchvision.transforms import transforms
# from resnet import *
import torch.optim as opt
from torchvision.models.quantization.resnet import resnet18 as qres18



def myresnet18():
    model = qres18()
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    model.fc = nn.Linear(512, 10)
    return model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossFunction = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = lossFunction(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train epoch : {}: [{}/{}]\t Loss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossFunction = nn.CrossEntropyLoss(reduction='sum')
    timelist = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        torch.cuda.synchronize()
        time_start = time.time()
        # print('here?  ---yes!')
        output = model(data)
        # print('here?  No!!')
        time_end = time.time()
        cur = time_end - time_start
        timelist = cur + timelist
        test_loss += lossFunction(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.9f},\n Accuracy: {:.2f}%,\ninference time: {:.4f}\n'.format(
        test_loss, 100. * correct / len(test_loader.dataset), timelist
    ))


def prepare_dataloader(num_workers=1, train_batch_size=64, test_batch_size=64):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )
    return train_loader, test_loader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_batch_size = 64
    test_batch_size = 64
    lr = 0.005
    momentum = 0.5
    epochs = 30
    save_model = True
    is_qat = True
    torch.manual_seed(1)
    train_loader, test_loader = prepare_dataloader(train_batch_size, test_batch_size)

    # model = RestNet18().to(device)
    model = myresnet18()


    if is_qat:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model.fuse_model()
        prepared_model = torch.quantization.prepare_qat(model, inplace=False)
        optimizer = opt.SGD(prepared_model.parameters(), lr=lr, momentum=momentum)
        if os.path.exists("ckpt/qcifar.pt"):
            prepared_model.load_state_dict(torch.load("ckpt/qcifar.pt", map_location="cuda"))
            # print(prepared_model)
            print('model state parameters load  successfully!!')


        for epoch in range(1, 2):
            train(prepared_model.to(device), device, train_loader, optimizer, epoch)
            print('finish training!!!!!!\n')
            prepared_model.to('cpu')
            convert_model = torch.quantization.convert(prepared_model.eval(), inplace=False)
            test(convert_model.to('cpu'), device='cpu', test_loader=test_loader)
        if save_model:
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            torch.save(prepared_model.state_dict(), 'ckpt/qcifar.pt')

        # print(prepared_model,'\n\n\n\n\n\n\n')

        # print('\n\n\n\n\n\n\n\n\n\n',convert_model)

        # for epoch in range(1, epochs + 1):
        #     train(convert_model, device, train_loader, optimizer, epoch)
        # print('qat finishing training !!')




    else:
        optimizer = opt.SGD(model.parameters(), lr=lr, momentum=momentum)
        if os.path.exists("ckpt/cifar.pt"):
            model.load_state_dict(torch.load("ckpt/cifar.pt", map_location="cuda"))
            print('model state parameters load to train_loader successfully!!')

        for epoch in range(1, epochs + 1):
            train(model.to(device), device, train_loader, optimizer, epoch)
            print('finish training!!!!!!\n')

            test(model.to('cpu'), device='cpu', test_loader=test_loader)
        if save_model:
            if not os.path.exists('ckpt'):
                os.makedirs('ckpt')
            torch.save(model.state_dict(), 'ckpt/cifar.pt')
