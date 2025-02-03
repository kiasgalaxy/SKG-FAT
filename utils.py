
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from tinyimagenet.tinyimagenet import TinyImageNet

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(dir_, train=True, transform=train_transform, download=False)
    test_dataset = datasets.CIFAR10(dir_, train=False, transform=test_transform, download=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_loaders_CIFAR100(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR100(dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(dir_, train=False, transform=test_transform, download=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_all_loaders_CIFAR10(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.Pad(padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=50000,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_all_loaders_CIFAR100(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.Pad(padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 0
    train_dataset = datasets.CIFAR100(dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(dir_, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=50000,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def get_loaders_TinyIN(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    trainset = TinyImageNet(dir_, 'train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = TinyImageNet(dir_, 'val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    return trainloader, testloader


def New_ImageNet_get_all_loaders_64(dir_, batch_size):
    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])
    num_workers = 0
    trainset = TinyImageNet(dir_, 'train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100000, shuffle=True, num_workers=num_workers)
    testset = TinyImageNet(dir_, 'val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def get_loaders_IN100(dir_, batch_size):
    traindir = os.path.join(dir_, 'train')
    valdir = os.path.join(dir_, 'val')
    testdir = os.path.join(dir_, 'test')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    trainset = datasets.ImageFolder(traindir, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valset = datasets.ImageFolder(valdir, transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) 
    return trainloader, valloader


def normalize(X):
    return (X - mu) / std


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon=(8 / 255.) / std):
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(tqdm(test_loader)):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


def class_wise_evaluate_pgd(test_loader, model, attack_iters, restarts, epsilon=(8 / 255.) / std):
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0

    PGD_loss_store_list_epoch = torch.zeros(10)
    count = 0

    model.eval()
    for i, (X, y) in enumerate(tqdm(test_loader)):
        count = count + 1

        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))

            PGD_loss_store_list_batch = torch.zeros(10)
            loss_list = torch.zeros(len(y)).cuda()
            for index in range(len(y)):
                loss_list[index] = F.cross_entropy(output[index], y[index])
            if X.shape[0] == 128:
                for class_idx in range(10):
                    class_indices = (class_idx == y).nonzero().squeeze()
                    class_loss = loss_list[class_indices]
                    # class_loss_avg = class_loss.sum() / len(class_loss)

                    if class_loss.size() == torch.Size([]):
                        class_loss_avg = 0
                    else:
                        class_loss_avg = class_loss.sum() / len(class_loss)

                    PGD_loss_store_list_batch[class_idx] = class_loss_avg
                PGD_loss_store_list_epoch = PGD_loss_store_list_epoch + PGD_loss_store_list_batch
            loss = loss_list.sum() / len(y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n, PGD_loss_store_list_epoch / (count-1)


def evaluate_powerful_pgd(test_loader, model, attack_iters, restarts, epsilon=(8 / 255.) / std):
    print(epsilon)
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):

        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(normalize(X + pgd_delta))
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        if i == 40:
            return pgd_loss / n, pgd_acc / n
    return pgd_loss / n, pgd_acc / n


def attack_fgsm(model, X, y, epsilon, alpha, restarts):
    attack_iters = 1
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_fgsm(test_loader, model, restarts,epsilon):
    # epsilon = (8 / 255.) / std
    alpha = epsilon # (8 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_fgsm(model, X, y, epsilon, alpha, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n


def class_wise_evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0

    count = 0
    test_loss_store_list_epoch = torch.zeros(10)

    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(test_loader)):
            count = count + 1

            X, y = X.cuda(), y.cuda()
            output = model(X)

            test_loss_store_list_batch = torch.zeros(10)
            loss_list = torch.zeros(len(y)).cuda()
            for index in range(len(y)):
                loss_list[index] = F.cross_entropy(output[index], y[index])
            if X.shape[0] == 128:
                for class_idx in range(10):
                    class_indices = (class_idx == y).nonzero().squeeze()

                    class_loss = loss_list[class_indices]
                    if class_loss.size() == torch.Size([]):
                        class_loss_avg = 0
                    else:
                        class_loss_avg = class_loss.sum() / len(class_loss)

                    # class_loss_avg = class_loss.sum() / len(class_loss)

                    test_loss_store_list_batch[class_idx] = class_loss_avg
                test_loss_store_list_epoch = test_loss_store_list_epoch + test_loss_store_list_batch

            loss = loss_list.sum() / len(y)

            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n, test_loss_store_list_epoch / (count - 1)


def CW_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()

    loss_value = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))
    return loss_value.mean()


def cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    # y_true = np.eye(10)[y.cuda().data.cpu().numpy()]
    # y_true = torch.from_numpy(y_true).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)

            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = CW_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd_cw(test_loader, model, attack_iters, restarts, epsilon_v):
    alpha = (2 / 255.) / std
    epsilon=(epsilon_v / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pgd_delta = cw_Linf_attack(model, X, y, epsilon, alpha, attack_iters=attack_iters, restarts=restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return pgd_loss / n, pgd_acc / n


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

