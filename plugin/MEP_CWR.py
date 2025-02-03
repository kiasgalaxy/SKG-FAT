import sys
import argparse
import copy
import logging
import os
import time
from utils import *
from torch.nn import functional as F
from torch.autograd import Variable
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='Dataset', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--epochs_reset', default=40, type=int)
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=8, type=float, help='Step size')
    parser.add_argument('--out-dir', default='Results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--factor', default=0.6, type=float, help='Label Smoothing')
    parser.add_argument('--lamda', default=2, type=float, help='Label Smoothing')
    parser.add_argument('--momentum_decay', default=0.3, type=float, help='momentum_decay')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'], help='Perturbation initialization method')
    return parser.parse_args()


args = get_args()
if args.data == 'CIFAR10':
    from models.CIFAR10.resnet import ResNet18
    num_classes = 10
elif args.data == 'CIFAR100':
    from models.CIFAR100.resnet import ResNet18
    num_classes = 100
else:
    raise ValueError


upper_limit_y = 1
lower_limit_y = 0
epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std


def _label_smoothing(label, factor):
    one_hot = np.eye(num_classes)[label.cuda().data.cpu().numpy()]
    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(num_classes - 1))
    return result


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


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

def filter_and_append_indices(arr, condition):
    result_indices = [i for i, element in enumerate(arr) if condition(element)]
    return result_indices

def find_common_elements(arr1, arr2):
    common_elements = [element for element in arr1 if element in arr2]
    return common_elements
    
def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, f'MEP_CWR')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logfile = os.path.join(output_path, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.data == 'CIFAR10':
        train_loader, test_loader = get_all_loaders_CIFAR10(args.data_dir,args.batch_size)
        model = ResNet18(num_classes)
    elif args.data == 'CIFAR100':
        train_loader, test_loader = get_all_loaders_CIFAR100(args.data_dir, args.batch_size)
        model = ResNet18()
    else:
        raise ValueError

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    num_of_example = 50000
    batch_size = args.batch_size
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    lr_steps = args.epochs * iter_num
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100/110, lr_steps * 105 / 110], gamma=0.1)

    # Training
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    for i, (X, y) in enumerate(train_loader):
        cifar_x, cifar_y = X.cuda(), y.cuda()
    import random
    def atta_aug(input_tensor, rst):
        batch_size = input_tensor.shape[0]
        x = torch.zeros(batch_size)
        y = torch.zeros(batch_size)
        flip = [False] * batch_size

        for i in range(batch_size):
            flip_t = bool(random.getrandbits(1))
            x_t = random.randint(0, 8)
            y_t = random.randint(0, 8)

            rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
            if flip_t:
                rst[i] = torch.flip(rst[i], [2])
            flip[i] = flip_t
            x[i] = x_t
            y[i] = y_t
        return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

    for epoch in range(args.epochs):
        CWlogger = CW_log()
        batch_size = args.batch_size
        cur_order = np.random.permutation(num_of_example)
        iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
        batch_idx = -batch_size
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        if epoch % args.epochs_reset == 0:
            temp=torch.rand(50000,3,32,32)
            if args.delta_init != 'previous':
                all_delta = torch.zeros_like(temp).cuda()
                all_momentum=torch.zeros_like(temp).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)
        idx = torch.randperm(cifar_x.shape[0])

        cifar_x =cifar_x[idx, :,:,:].view(cifar_x.size())
        cifar_y = cifar_y[idx].view(cifar_y.size())
        all_delta=all_delta[idx, :, :, :].view(all_delta.size())
        all_momentum=all_momentum[idx, :, :, :].view(all_delta.size())
        for i in range(iter_num):
            batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
            X=cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            y= cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            delta =all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            momentum=all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
            X=X.cuda()
            y=y.cuda()
            batch_size = X.shape[0]
            rst = torch.zeros(batch_size, 3, 32, 32).cuda()
            X, transform_info = atta_aug(X, rst)
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).cuda()).float()

            delta.requires_grad = True
            ori_output = model(X + delta[:X.size(0)])
            ori_loss = LabelSmoothLoss(ori_output, label_smoothing.float())

            decay=args.momentum_decay
            ori_loss.backward(retain_graph=True)
            x_grad = delta.grad.detach()
            grad_norm = torch.norm(x_grad, p=1)
            momentum = x_grad/grad_norm+momentum * decay

            next_delta.data = clamp(delta + alpha * torch.sign(momentum), -epsilon, epsilon)
            next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)

            delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

            delta = delta.detach()
            output = model(X + delta[:X.size(0)])
            clean_output = model(X)
            CWlogger.update_robust(output, y)
            CWlogger.update_clean(clean_output, y)
            loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
            loss = LabelSmoothLoss(output, (label_smoothing).float())
            for i in range(10):
                class_indices = (i == y).nonzero().squeeze()
                if class_indices.numel() > 0:
                    C_output = clean_output[class_indices]
                    R_output = output[class_indices]
                    ClassRegLoss = loss_fn(R_output, C_output)
                    loss += args.lamda * ClassRegLoss

            
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum
            all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]=next_delta
            
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n)

        if args.data == 'CIFAR10':
            model_test = ResNet18(num_classes)
        elif args.data == 'CIFAR100':
            model_test = ResNet18()
        else:
            raise ValueError
        model_test = torch.nn.DataParallel(model_test)
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        if best_result <= pgd_acc:
            best_result = pgd_acc
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))

    torch.save(model.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)

if __name__ == "__main__":
    main()
