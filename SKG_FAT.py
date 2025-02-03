
from __future__ import print_function
import os
import time
import argparse
import logging
import torchvision
from utils import *
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
logger = logging.getLogger('logger')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR100', choices=['CIFAR10', 'CIFAR100', 'TinyIN', 'IN100'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--data-dir', default='Dataset', type=str)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--out-dir', default='Results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--lamda', default=1.0, type=float, help='Penalize regularization term')
    parser.add_argument('--begin', default=1, type=int)
    parser.add_argument('--LSType', default='SKLR', type=str, choices=['SKLR', 'SLS'])
    parser.add_argument('--LS_max', default=0.9, type=float)
    parser.add_argument('--LS_min', default=0.3, type=float)
    parser.add_argument('--RegType', default='CWR', type=str, choices=['AGR', 'CWR', 'Normal'])
    arguments = parser.parse_args()
    return arguments


args = get_args()

output_path = os.path.join(args.out_dir, f'SKG_{args.data}_RegType_{args.RegType}_Lam_{args.lamda}')
if not os.path.exists(output_path):
    os.makedirs(output_path)
logfile = os.path.join(output_path, f'train_info.log')
if os.path.exists(logfile):
    os.remove(logfile)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(output_path, f'train_info.log'))

logger.info(args)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std


if args.data == 'CIFAR10':
    from models.CIFAR10.resnet import ResNet18
    num_classes = 10
    args.DataDir = 'Dataset'
    args.epochs = 110
elif args.data == 'CIFAR100':
    from models.CIFAR100.resnet import ResNet18
    num_classes = 100
    args.DataDir = 'Dataset'
    args.epochs = 110
elif args.data == 'TinyIN':
    from models.TinyIN.resnet import ResNet18
    num_classes = 200
    args.DataDir = ''
    args.epochs = 110
elif args.data == 'IN100':
    num_classes = 100
    args.DataDir = ''
    args.epochs = 50
else:
    raise ValueError

decay_arr = [0, args.epochs - 10, args.epochs - 5, args.epochs]

if args.data == 'CIFAR10':
    train_loader, test_loader = get_loaders(args.DataDir, args.batch_size)
    model = ResNet18(num_classes)
elif args.data == 'CIFAR100':
    train_loader, test_loader = get_loaders_CIFAR100(args.DataDir, args.batch_size)
    model = ResNet18()
elif args.data == 'TinyIN':
    train_loader, test_loader = get_loaders_TinyIN(args.DataDir, args.batch_size)
    model = ResNet18()
elif args.data == 'IN100':
    train_loader, test_loader = get_loaders_IN100(args.DataDir, args.batch_size)
    model = torchvision.models.resnet18(num_classes=num_classes)
else:
    raise ValueError


target_model = model.cuda()
target_model.train()

opt = torch.optim.SGD(target_model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
lr_steps = decay_arr[3] * len(train_loader)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * decay_arr[1]/decay_arr[3], lr_steps * decay_arr[2] / decay_arr[3]], gamma=0.1)


def _label_smoothing(label, factor, num_classes):
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


def class_label_smoothing(y, factors):
    class_num = len(factors)
    one_hot = F.one_hot(y, class_num).float()
    batch_smooth_factors = factors[y].view(-1, 1)
    main_matrix = one_hot * batch_smooth_factors
    result = main_matrix + (one_hot - 1.) * (batch_smooth_factors - 1)/float(class_num - 1)
    return result


def upper_lower(original_tensor, a, b):
    clipped_tensor = torch.clip(original_tensor, b, a)
    return clipped_tensor


def find_common_elements(arr1, arr2):
    common_elements = [element for element in arr1 if element in arr2]
    return common_elements


def filter_and_append_indices(arr, condition):
    result_indices = [i for i, element in enumerate(arr) if condition(element)]
    return result_indices


class CW_log():
    def __init__(self, class_num=num_classes) -> None:
        self.N = 0
        self.robust_acc = 0
        self.clean_acc = 0
        self.cw_robust = torch.zeros(num_classes).cuda()
        self.cw_clean = torch.zeros(num_classes).cuda()
        self.losslog = torch.zeros(6).cuda()
        self.class_num = class_num
        self.count = 0
        self.counter = 0

    def update_clean(self, output, y):
        self.N += len(output)
        pred = output.max(1)[1]
        correct = pred == y
        self.clean_acc += correct.sum()
        for i, c in enumerate(y):
            if correct[i]:
                self.cw_clean[c] += 1

    def update_robust(self, output, y):
        pred = output.max(1)[1]
        correct = pred == y
        self.robust_acc += correct.sum()
        for i, c in enumerate(y):
            if correct[i]:
                self.cw_robust[c] += 1

    def update_loss(self, new_loss):
        if self.count <= 5:
            self.losslog[self.count] += new_loss
            self.count += 1
        else:
            self.count = 0
            self.losslog[self.count] += new_loss
            self.count += 1
        self.counter += 1

    def result(self):
        N = self.N
        m = self.class_num
        return self.clean_acc / N, self.robust_acc / N, m * self.cw_clean / N, m * self.cw_robust / N, self.losslog / self.counter * 6


def train(args, model, train_loader, opt, scheduler, epoch, init_momentum):
    epoch_time = 0
    train_loss = 0
    train_loss_normal = 0
    train_acc = 0
    train_n = 0
    train_logger = CW_log()

    delta = init_momentum

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        batch_start_time = time.time()

        if x.shape[0] == args.batch_size:
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.LS_max, num_classes)).cuda())
            delta.requires_grad = True
            ori_output = model(x + delta)

            ori_loss = torch.nn.CrossEntropyLoss()(ori_output, label_smoothing.float())
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()

            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - x, upper_limit - x)
            delta = delta.detach()

            # update
            init_momentum = clamp(delta, lower_limit - x, upper_limit - x)

            logits = ori_output
            output = model(x + delta)
            clean_output = model(x)
            loss_robust = (1.0 / args.batch_size) * torch.sum(torch.sum(torch.square(torch.sub(logits, output)), dim=1))
            loss = LabelSmoothLoss(output, label_smoothing.float()) + float(args.lamda) * loss_robust

            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            # Total training loss in this batch
            train_loss = train_loss + loss.item() * y.size(0)
            # Clean loss in this batch
            train_loss_normal += ori_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_logger.update_robust(output, y)
            train_logger.update_clean(clean_output, y)

        batch_end_time = time.time()
        epoch_time += batch_end_time - batch_start_time

    lr = scheduler.get_lr()[0]
    logger.info('Epoch \t Seconds \t LR \tTrain normal Loss  \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f\t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss_normal/train_n, train_loss / train_n, train_acc / train_n)
    return train_loss/train_n, train_loss_normal/train_n, init_momentum, epoch_time, train_logger.result()


def train_CWR(args, model, train_loader, opt, scheduler, epoch, init_momentum, acc_factors):
    epoch_time = 0
    train_loss = 0
    train_loss_normal = 0
    train_acc = 0
    train_n = 0

    delta = init_momentum
    train_logger = CW_log()

    for _, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        batch_start_time = time.time()

        if x.shape[0] == args.batch_size:
            if args.LSType == 'SKLR':
                acc_factors = torch.abs(torch.log(acc_factors))
                acc_factors = upper_lower(acc_factors, args.LS_max, args.LS_min)
                label_smoothing = Variable(class_label_smoothing(y, acc_factors)).cuda().float()
            elif args.LSType == 'SLS':
                label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.LS_max, num_classes)).cuda())
            else:
                raise ValueError
            delta.requires_grad = True
            ori_output = model(x + delta)

            ori_loss = torch.nn.CrossEntropyLoss()(ori_output, label_smoothing.float())
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()

            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - x, upper_limit - x)
            delta = delta.detach()

            # update
            init_momentum = clamp(delta, lower_limit - x, upper_limit - x)
            output = model(x + delta)
            clean_output = model(x)
            loss = LabelSmoothLoss(output, label_smoothing.float())

            loss_fn = torch.nn.MSELoss(reduction='mean')
            for i in range(num_classes):
                class_indices = (i == y).nonzero().squeeze()
                if class_indices.numel() > 0:
                    C_output = clean_output[class_indices]
                    R_output = output[class_indices]
                    ClassRegLoss = loss_fn(C_output, R_output)
                    loss += args.lamda * acc_factors[i] * ClassRegLoss

            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            train_loss = train_loss + loss.item() * y.size(0)
            train_loss_normal += ori_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_logger.update_robust(output, y)
            train_logger.update_clean(clean_output, y)

        batch_end_time = time.time()
        epoch_time += batch_end_time - batch_start_time

    lr = scheduler.get_lr()[0]
    logger.info('Epoch \t Seconds \t LR \tTrain normal Loss  \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f\t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss_normal/train_n, train_loss / train_n, train_acc / train_n)
    return train_loss/train_n, train_loss_normal/train_n, init_momentum, epoch_time, train_logger.result()


def train_AGR(args, model, train_loader, opt, scheduler, epoch, init_momentum, GCGR, GCBR, BCGR, BCBR, acc_factors):
    epoch_time = 0
    train_loss = 0
    train_loss_normal = 0
    train_acc = 0
    train_n = 0

    delta = init_momentum
    train_logger = CW_log()

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        batch_start_time = time.time()

        GCGR_index = torch.where(torch.isin(y, GCGR))[0].cuda()
        GCBR_index = torch.where(torch.isin(y, GCBR))[0].cuda()
        BCGR_index = torch.where(torch.isin(y, BCGR))[0].cuda()
        BCBR_index = torch.where(torch.isin(y, BCBR))[0].cuda()

        if x.shape[0] == args.batch_size:
            if args.LSType == 'SKLR':
                acc_factors = torch.abs(torch.log(acc_factors))
                acc_factors = upper_lower(acc_factors, args.LS_max, args.LS_min)
                label_smoothing = Variable(class_label_smoothing(y, acc_factors)).cuda().float()
            elif args.LSType == 'SLS':
                label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.LS_max, num_classes)).cuda())
            else:
                raise ValueError
            delta.requires_grad = True
            ori_output = model(x + delta)

            ori_loss = torch.nn.CrossEntropyLoss()(ori_output, label_smoothing.float())
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()

            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - x, upper_limit - x)
            delta = delta.detach()

            # update
            init_momentum = clamp(delta, lower_limit - x, upper_limit - x)
            output = model(x + delta)
            clean_output = model(x)

            loss_fn = torch.nn.MSELoss(reduction='mean')
            loss = LabelSmoothLoss(output, label_smoothing.float())

            if len(GCGR_index):
                GCGR_Routput = torch.index_select(output, 0, GCGR_index)
                GCGR_Coutput = torch.index_select(clean_output, 0, GCGR_index)
                GCGRloss = loss_fn(GCGR_Routput, GCGR_Coutput)
                loss += GCGRloss * args.lamda
            if len(GCBR_index):
                GCBR_Routput = torch.index_select(output, 0, GCBR_index)
                GCBR_Coutput = torch.index_select(clean_output, 0, GCBR_index)
                GCBRloss = loss_fn(GCBR_Routput, GCBR_Coutput)
                loss += GCBRloss * args.lamda
            if len(BCGR_index):
                BCGR_Routput = torch.index_select(output, 0, BCGR_index)
                BCGR_Coutput = torch.index_select(clean_output, 0, BCGR_index)
                BCGRloss = loss_fn(BCGR_Routput, BCGR_Coutput)
                loss += BCGRloss * args.lamda
            if len(BCBR_index):
                BCBR_Routput = torch.index_select(output, 0, BCBR_index)
                BCBR_Coutput = torch.index_select(clean_output, 0, BCBR_index)
                BCBRloss = loss_fn(BCBR_Routput, BCBR_Coutput)
                loss += BCBRloss * args.lamda

            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            train_loss = train_loss + loss.item() * y.size(0)
            train_loss_normal += ori_loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_logger.update_robust(output, y)
            train_logger.update_clean(clean_output, y)

        batch_end_time = time.time()
        epoch_time += batch_end_time - batch_start_time

    lr = scheduler.get_lr()[0]
    logger.info('Epoch \t Seconds \t LR \tTrain normal Loss  \t Train Loss \t Train Acc')
    logger.info('%d \t %.1f \t %.4f\t %.4f \t %.4f \t %.4f',
                epoch, epoch_time, lr, train_loss_normal/train_n, train_loss / train_n, train_acc / train_n)
    return train_loss/train_n, train_loss_normal/train_n, init_momentum, epoch_time, train_logger.result()


def main():
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    epoch_loss_list = []
    epoch_loss_ori_list = []

    Good_condition_function = lambda x: x > 0
    Bad_condition_function = lambda x: x < 0

    if args.data == 'CIFAR10':
        perturbation_init = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    elif args.data == 'CIFAR100':
        perturbation_init = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    elif args.data == 'TinyIN':
        perturbation_init = torch.zeros(args.batch_size, 3, 64, 64).cuda()
    elif args.data == 'IN100':
        perturbation_init = torch.zeros(args.batch_size, 3, 224, 224).cuda()
    else:
        raise ValueError

    for j in range(len(epsilon)):
        perturbation_init[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    perturbation_init = clamp(alpha * torch.sign(perturbation_init), -epsilon, epsilon)

    for epoch in range(args.epochs):
        if epoch < args.begin:
            train_loss, train_loss_nor, perturbation_init, epoch_time, train_log = train(args, target_model, train_loader, opt, scheduler, epoch, perturbation_init)
        else:
            Avg_Cleanacc = train_log[0]
            Avg_Robustacc = train_log[1]
            Class_Cleanacc = train_log[2]
            Class_Robustacc = train_log[3]

            diffC = Class_Cleanacc - Avg_Cleanacc
            diffR = Class_Robustacc - Avg_Robustacc

            Good_CleanClass = filter_and_append_indices(diffC, Good_condition_function)
            Bad_CleanClass = filter_and_append_indices(diffC, Bad_condition_function)
            Good_RobustClass = filter_and_append_indices(diffR, Good_condition_function)
            Bad_RobustClass = filter_and_append_indices(diffR, Bad_condition_function)
            GCGR = find_common_elements(Good_CleanClass, Good_RobustClass)
            GCBR = find_common_elements(Good_CleanClass, Bad_RobustClass)
            BCGR = find_common_elements(Bad_CleanClass, Good_RobustClass)
            BCBR = find_common_elements(Bad_CleanClass, Bad_RobustClass)
            GCGR, GCBR, BCGR, BCBR = torch.tensor(GCGR).cuda(), torch.tensor(GCBR).cuda(), torch.tensor(BCGR).cuda(), torch.tensor(BCBR).cuda()

            if args.RegType == 'AGR':
                train_loss, train_loss_nor, perturbation_init, epoch_time, train_log = train_AGR(args, target_model, train_loader, opt, scheduler, epoch, perturbation_init, GCGR, GCBR, BCGR, BCBR, Class_Cleanacc)
            elif args.RegType == 'CWR':
                train_loss, train_loss_nor, perturbation_init, epoch_time, train_log = train_CWR(args, target_model, train_loader, opt, scheduler, epoch, perturbation_init, Class_Cleanacc)
            else:
                train_loss, train_loss_nor, perturbation_init, epoch_time, train_log = train(args, target_model, train_loader, opt, scheduler, epoch, perturbation_init)

        if args.data == 'IN100':
            model_test = torchvision.models.resnet18(num_classes=num_classes)
            model_test.cuda()
        elif args.data == 'CIFAR10':
            model_test = ResNet18(num_classes).cuda()
        else:
            model_test = ResNet18().cuda()
        model_test.load_state_dict(target_model.state_dict())
        model_test.float()
        model_test.eval()
        adv_loss, adv_acc = evaluate_pgd(test_loader, model_test, 10, 1, epsilon)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(adv_acc)
        epoch_loss_list.append(train_loss)
        epoch_loss_ori_list.append(train_loss_nor)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, adv_loss, adv_acc)
        if best_result <= adv_acc:
            best_result = adv_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))

    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    logger.info(epoch_loss_list)
    logger.info(epoch_loss_ori_list)

main()

