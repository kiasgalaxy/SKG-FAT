import os
import torch.nn as nn
from Eval_Results.autoattack import AutoAttack
from torchattacks import  MIFGSM, APGD
from utils import *
import argparse
import logging
from models.resnet import ResNet18
import tqdm

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='Dataset')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_name', type=str, default='109')
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', type=str, default='Eval_Results/Results')

    parser.add_argument('--save_file_name', type=str)
    arguments = parser.parse_args()
    return arguments

args = get_args()
num_classes = 10

class CW_log():
    def __init__(self, class_num = num_classes) -> None:
        self.N = 0
        self.robust_acc = 0
        self.clean_acc = 0
        self.cw_robust = torch.zeros(num_classes).cuda()
        self.cw_clean = torch.zeros(num_classes).cuda()
        self.class_num = class_num
    
    def update_clean(self, output, y):
        # self.N += len(output)
        pred = output.max(1)[1]
        correct = pred == y
        self.clean_acc += correct.sum()

        for i, c in enumerate(y):
            if correct[i]:
                self.cw_clean[c] += 1
    
    def update_robust(self, output, y):
        self.N += len(output)
        pred = output.max(1)[1]
        correct = pred == y
        self.robust_acc += correct.sum()

        for i, c in enumerate(y):
            if correct[i]:
                self.cw_robust[c] += 1
    
    def result(self):
        N = self.N
        m = self.class_num
        return self.clean_acc/N, self.robust_acc/N, m*self.cw_clean/N, m*self.cw_robust/N
    
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
logfile1 = os.path.join(args.out_dir, f'AA_{args.save_file_name}.txt')

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=os.path.join(args.out_dir, f'Evaluate_{args.save_file_name}.log'))

full_path = args.model_path + args.model_name + '.pth'
target_model = ResNet18(num_classes)
target_model.load_state_dict(torch.load(full_path))
target_model.cuda().eval()

train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std

def pgd_loss(model, x, y, eps, alpha, n_iters=10, restarts = 1):
    delta = attack_pgd(model, x, y, eps, alpha, n_iters, restarts)
    with torch.no_grad():
        robust_output = model(x + delta)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(robust_output, y)
    return loss, robust_output.clone().detach()

def attack_cw(model, x, y, eps, alpha, n_iters = 20):
    delta = cw_Linf_attack(model, x, y, eps, alpha, n_iters, restarts = 1)
    with torch.no_grad():
        robust_output = model(x + delta)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(robust_output, y)
    return loss, robust_output.clone().detach()

def eval_pgd10(model, loader, eps, alpha, n_iters = 10):
    print('attack method : PGD10')
    model.eval()
    logger = CW_log()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        _, output = pgd_loss(model,x,y,eps,alpha,n_iters = 10)
        logger.update_robust(output, y)

        clean_output = model(x).detach()
        logger.update_clean(clean_output, y)
    print('PGD10 attack finished')
    return logger.result()

def eval_pgd20(model, loader, eps, alpha, n_iters = 20):
    print('attack method : PGD20')
    model.eval()
    logger = CW_log()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        _, output = pgd_loss(model,x,y,eps,alpha,n_iters = 20)
        logger.update_robust(output, y)
    print('PGD20 attack finished')
    return logger.result()

def eval_pgd50(model, loader, eps, alpha, n_iters = 50):
    print('attack method : PGD50')
    model.eval()
    logger = CW_log()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        _, output = pgd_loss(model,x,y,eps,alpha,n_iters = 50)
        logger.update_robust(output, y)
    print('PGD50 attack finished')
    return logger.result()

def eval_cw(model, loader, eps, alpha, n_iters = 20):
    print('attack method : CW')
    model.eval()
    logger = CW_log()
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        _, output = attack_cw(model,x,y,eps,alpha,n_iters = 20)
        logger.update_robust(output, y)
    print('CW attack finished')
    return logger.result()

def eval_MI_FGSM(model, loader, eps, alpha):
    print('attack method : MI-FGSM')
    model.eval()
    logger = CW_log()
    MIFGSM_attack = MIFGSM(model, eps, alpha, steps=10, decay=1.0)
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        adv_x = MIFGSM_attack(x, y)
        output = model(adv_x)
        logger.update_robust(output, y)
    print('MI-FGSM attack finished')
    return logger.result()

def eval_APGD(model, loader, eps):
    print('attack method : APGD')
    model.eval()
    logger = CW_log()
    APGD_attack = APGD(model, norm='Linf', eps = eps, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
    for batch_idx, batch in enumerate(loader):
        x, y = batch
        x, y = x.cuda(), y.cuda()
        adv_x = APGD_attack(x, y)
        output = model(adv_x)
        logger.update_robust(output, y)
    print('A-PGD attack finished')
    return logger.result()

def eval_AutoAttack(model, loader, eps):
    print('attack method : AutoAttack')
    model.eval()
    logger = CW_log()
    adversary1 = AutoAttack(model, norm=args.norm, eps=eps, version='standard',log_path=logfile1)
    l = [x for (x, y) in loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in loader]
    y_test = torch.cat(l, 0)
    adv_x = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],bs=args.batch_size)
    with torch.no_grad():
        n_batches = int(np.ceil(adv_x.shape[0] / args.batch_size))
        for batch_idx in range(n_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min((batch_idx + 1) * args.batch_size, adv_x.shape[0])
            advx = adv_x[start_idx:end_idx, :].clone().cuda()
            advy = y_test[start_idx:end_idx].clone().cuda()
            output = model(advx)
            logger.update_robust(output, advy)
    print(' AutoAttack finished')
    return logger.result()

logger.info(args.model_name)
logger.info('Clean \t MI-FGSM \t PGD-10 \t PGD-50 \t APGD \t CW \t AA \t Avg RA')
# logger.info('Clean \t PGD-10')

MI_FGSM_result = eval_MI_FGSM(target_model, test_loader, epsilon, alpha)
pgd10_result = eval_pgd10(target_model ,test_loader, epsilon, alpha)
pgd50_result = eval_pgd50(target_model ,test_loader, epsilon, alpha)
APGD_result = eval_APGD(target_model, test_loader, epsilon)
cw_result = eval_cw(target_model ,test_loader, epsilon, alpha)
AA_result = eval_AutoAttack(target_model, test_loader, epsilon)

clean_acc = pgd10_result[0]
MI_FGSM_acc = MI_FGSM_result[1]
pgd_10_acc = pgd10_result[1]
pgd_50_acc = pgd50_result[1]
APGD_acc = APGD_result[1]
cw_acc = cw_result[1]
AA_acc = AA_result[1]

logger.info(args.model_path)
logger.info('%.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', clean_acc, MI_FGSM_acc, pgd_10_acc, pgd_50_acc, APGD_acc, cw_acc, AA_acc)
