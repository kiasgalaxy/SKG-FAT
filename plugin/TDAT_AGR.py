from __future__ import print_function
import os
import math
import torch
import argparse
import logging
from utils import *
import torch.nn as nn
from tqdm import tqdm

from torch.autograd import Variable
from torch.nn import functional as F
logger = logging.getLogger('logger')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--data-dir', default='Dataset', type=str)
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--epsilon', default=8., type=float, help='Perturbation budget')
    parser.add_argument('--alpha', default=10., type=float, help='Step size')

    # TDAT
    parser.add_argument('--gamma', default=0.05, type=float, help='Label relaxation factor')
    parser.add_argument('--beta', default=0.6, type=float)
    parser.add_argument('--lamda', default=0.05, type=float, help='Penalize regularization term')
    parser.add_argument('--batch-m', default=0.75, type=float)
    
    parser.add_argument('--out-dir', default='Results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arguments = parser.parse_args()
    return arguments


args = get_args()
if args.data == 'CIFAR10':
    from models.CIFAR10.resnet import ResNet18
    num_classes = 10
elif args.data == 'CIFAR100':
    from models.CIFAR100.resnet import ResNet18
    num_classes = 100
else:
    raise ValueError

output_path = os.path.join(args.out_dir, f'TDAT_AGR')
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
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
    model = ResNet18(num_classes)
elif args.data == 'CIFAR100':
    train_loader, test_loader = get_loaders_CIFAR100(args.data_dir, args.batch_size)
    model = ResNet18()
else:
    raise ValueError

target_model = model.cuda()
opt = torch.optim.SGD(target_model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
decay_arr = [0, 100, 105, 110]
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


def find_common_elements(arr1, arr2):
    common_elements = [element for element in arr1 if element in arr2]
    return common_elements


def filter_and_append_indices(arr, condition):
    result_indices = [i for i, element in enumerate(arr) if condition(element)]
    return result_indices


class CW_log():
    def __init__(self, class_num = num_classes) -> None:
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
        return self.clean_acc/N, self.robust_acc/N, m*self.cw_clean/N, m*self.cw_robust/N


def Atk_FGSM(model, x, y, eps, alpha, init_momentum):
    delta = init_momentum
    delta.requires_grad = True

    output = model(x + delta)
    loss = nn.CrossEntropyLoss()(output, y.float())
    loss.backward(retain_graph=True)
    grad = delta.grad.detach()
    d = torch.clamp(delta + alpha * torch.sign(grad), min=-eps, max=eps)
    d = torch.clamp(d, lower_limit - x, upper_limit - x)
    delta.data = d
    delta.grad.zero_()

    return delta.detach()


def perturbation_init(perturbation_size, epsilon, alpha):
    delta_init = torch.zeros(perturbation_size, 3, 32, 32).cuda()
    for j in range(len(epsilon)):
        delta_init[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    delta_init = clamp(alpha * torch.sign(delta_init), -epsilon, epsilon)
    return delta_init


def train_AGR(args, model, train_loader, opt, scheduler, epoch, init_momentum, GCGR, GCBR, BCGR, BCBR):
    train_loss = 0
    train_acc = 0
    train_n = 0
    train_logger = CW_log()

    # dynamic label relaxtion
    gamma = math.tan(1 - (epoch/args.epochs)) * args.beta
    if gamma < args.gamma:
        gamma = args.gamma

    for idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()

        GCGR_index = torch.where(torch.isin(y, GCGR))[0].cuda()
        GCBR_index = torch.where(torch.isin(y, GCBR))[0].cuda()
        BCGR_index = torch.where(torch.isin(y, BCGR))[0].cuda()
        BCBR_index = torch.where(torch.isin(y, BCBR))[0].cuda()

        if x.shape[0] == args.batch_size:
            label_smoothing_inner = Variable(torch.tensor(_label_smoothing(y, gamma, num_classes)).cuda())

            output = model(x + init_momentum)
            delta = Atk_FGSM(model, x, label_smoothing_inner, epsilon, alpha, init_momentum)

            # update
            init_momentum = args.batch_m * init_momentum + (1.0 - args.batch_m) * delta
            init_momentum = clamp(init_momentum, -epsilon, epsilon)
            init_momentum = clamp(delta, lower_limit - x, upper_limit - x)

            x_adv = x + delta
            clean_output = model(x)
            output_adv = model(x_adv)
            loss = nn.CrossEntropyLoss(label_smoothing=(1.0-gamma))(output, y)

            loss_fn = torch.nn.MSELoss(reduction='mean')
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
            train_acc += (output_adv.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_logger.update_robust(output, y)
            train_logger.update_clean(clean_output, y)

    lr = scheduler.get_lr()[0]
    logger.info('%d \t %.4f \t %.4f \t %.4f', epoch, lr, train_loss / train_n, train_acc / train_n)
    return train_loss/train_n, init_momentum, train_logger.result()



def train(args, model, train_loader, opt, scheduler, epoch, init_momentum):
    train_loss = 0
    train_acc = 0
    train_n = 0
    train_logger = CW_log()

    # dynamic label relaxtion
    gamma = math.tan(1 - (epoch/args.epochs)) * args.beta
    if gamma < args.gamma:
        gamma = args.gamma

    for idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.cuda(), y.cuda()
        if x.shape[0] == args.batch_size:
            label_smoothing_inner = Variable(torch.tensor(_label_smoothing(y, gamma, num_classes)).cuda())

            output = model(x + init_momentum)
            delta = Atk_FGSM(model, x, label_smoothing_inner, epsilon, alpha, init_momentum)

            # update
            init_momentum = args.batch_m * init_momentum + (1.0 - args.batch_m) * delta
            init_momentum = clamp(init_momentum, -epsilon, epsilon)
            init_momentum = clamp(delta, lower_limit - x, upper_limit - x)

            x_adv = x + delta
            clean_output = model(x)
            output_adv = model(x_adv)
            loss_adv = nn.CrossEntropyLoss(label_smoothing=(1.0-gamma))(output, y)

            # TD loss
            nat_probs = F.softmax(output, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
            loss_MSE = (1.0 / args.batch_size) * torch.sum(torch.sum(torch.square(torch.sub(output, output_adv)),dim=1) * torch.tanh(1.0000001 - true_probs))
            loss = loss_adv + args.lamda * loss_MSE

            opt.zero_grad()
            model.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            train_loss = train_loss + loss.item() * y.size(0)
            train_acc += (output_adv.max(1)[1] == y).sum().item()
            train_n += y.size(0)

            train_logger.update_robust(output, y)
            train_logger.update_clean(clean_output, y)

    lr = scheduler.get_lr()[0]
    logger.info('%d \t %.4f \t %.4f \t %.4f', epoch, lr, train_loss / train_n, train_acc / train_n)
    return train_loss/train_n, init_momentum, train_logger.result()


def main():
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []
    epoch_loss_list = []

    Good_condition_function = lambda x: x > 0
    Bad_condition_function = lambda x: x < 0

    init_momentum = perturbation_init(args.batch_size, epsilon, alpha)

    for epoch in range(args.epochs):
        if epoch < 1:
            train_loss, init_momentum, train_log = train(args, target_model, train_loader, opt, scheduler, epoch, init_momentum)
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

            train_loss, init_momentum, train_log = train_AGR(args, target_model, train_loader, opt, scheduler, epoch, init_momentum, GCGR, GCBR, BCGR, BCBR)

        if args.data == 'CIFAR10':
            model_test = ResNet18(num_classes)
        elif args.data == 'CIFAR100':
            model_test = ResNet18()
        else:
            raise ValueError
        model_test.cuda()
        model_test.load_state_dict(target_model.state_dict())
        model_test.float()
        model_test.eval()

        adv_loss, adv_acc = evaluate_pgd(test_loader, model_test, 10, 1, epsilon)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(adv_acc)
        epoch_loss_list.append(train_loss)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, adv_loss, adv_acc)
        if best_result <= adv_acc:
            best_result = adv_acc
            torch.save(target_model.state_dict(), os.path.join(output_path, 'best_model.pth'))
        torch.save(target_model.state_dict(), os.path.join(output_path, 'final_model.pth'))

    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    logger.info(epoch_loss_list)

main()

