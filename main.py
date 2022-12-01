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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):

    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:

            # print('here')
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end='')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')


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


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')

    cnt = 0
    for image, target in data_loader:
        start_time = time.time()

        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))
        avgloss.update(loss, image.size(0))
        if cnt >= ntrain_batches:
            print('Loss: ', avgloss.avg)

            print('Training: * Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))
            return

    print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=top1, top5=top5))
    print('finish training! ')
    return


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    lossFunction = nn.CrossEntropyLoss(reduction='sum')
    timelist = 0
    for data, target in test_loader:
        data, target = data.to("cpu"), target.to("cpu")
        torch.cuda.synchronize()
        time_start = time.time()
        # print('here')
        output = model(data).to("cpu")

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

if __name__ == '__main__':
    model = myresnet18()
    device = torch.device('cuda' if not torch.cuda.is_available() else 'cpu')
    train_batch_size = 16
    test_batch_size = 16
    lr = 0.01
    momentum = 0.5
    epochs = 15
    save_model = True
    is_qat = True
    # saved_model_dir = 'ckpt/'
    # scripted_float_model_file = 'orignal.pth'
    torch.manual_seed(1)
    train_loader, test_loader = prepare_dataloader(train_batch_size, test_batch_size)
    criterion = nn.CrossEntropyLoss()


    model.train()
    model.fuse_model()

    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    torch.quantization.prepare_qat(model, inplace=True)
    # print(model)
    # sys.exit(1)

    num_train_batches = 20
    num_eval_batches = 100

    # if os.path.exists('main/qfuck.pt'):
    #     prepared_model.load_state_dict(torch.load("main/qfuck.pt", map_location="cuda"))
    #     print('model state parameters load to test_loader successfully!!')
    # else:
    #     for nepoch in range(8):
    #         train_one_epoch(prepared_model.to(device), criterion, optimizer, train_loader, torch.device(device), num_train_batches)
    #
    #     if not os.path.exists('main'):
    #         os.makedirs('main')
    #     torch.save(prepared_model.state_dict(), 'main/qfuck.pt')
    # convert_model = torch.quantization.convert(prepared_model.eval(), inplace=True)
    # print(convert_model)
    #
    # top1, top5 = evaluate(convert_model, criterion, test_loader, neval_batches=test_batch_size)
    # print('Epoch %d :Evaluation accuracy on %d images, %2.2f' % (epochs, num_eval_batches * test_batch_size, top1.avg))

    # test(convert_model,device,test_loader)
    num_train_batches = 50

    # QAT takes time and one needs to train over a few epochs.
    # Train and check accuracy after each epoch

    for nepoch in range(8):
        train_one_epoch(model, criterion, optimizer, train_loader,device, num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        # if not os.path.exists('main'):
        #     os.makedirs('main')
        # torch.save(model.state_dict(), 'main/qfuck.pt')
        #
        # # Check the accuracy after each epoch
        #
        # model.load_state_dict(torch.load("main/qfuck.pt", map_location="cpu"))
        # print('model state parameters load to test_loader successfully!!\n')
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model, criterion, test_loader, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f' % (
        nepoch, num_eval_batches * 50, top1.avg))