import argparse
import copy
import logging
import os
import time

import numpy as np
import torch

import torch
import torch.nn as nn
from ImageNet_models import *
import torch.nn.functional as F
from utils02 import (upper_limit, lower_limit, std, clamp, get_loaders, ImageNet_get_loaders,New_ImageNet_get_loaders_64,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


m_sample_list = []
all_block_count = 0

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self,prob, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        identity = x.clone()
        global m_sample_list
        global all_block_count
        if self.training:
            if len(m_sample_list) != 8:
                all_block_count = all_block_count + 1
                m_sample = self.m.sample()
                m_sample_list.append(m_sample)
                if torch.equal(m_sample, torch.ones(1)):
                    print("******************")
                    self.conv1.weight.requires_grad = True
                    self.conv2.weight.requires_grad = True
                    out = F.relu(self.bn1(x))
                    shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
                    out = self.conv1(out)
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += shortcut
                else:
                    print("!!!!!!!!!!!!!!!!!!")
                    self.conv1.weight.requires_grad = False
                    self.conv2.weight.requires_grad = False

                    out = F.relu(self.bn1(x))
                    out = self.shortcut(out) if hasattr(self, 'shortcut') else x
                return out
            else:

                if torch.equal(m_sample_list[8 - all_block_count], torch.ones(1)):
                    all_block_count = all_block_count - 1
                    print("******************")
                    self.conv1.weight.requires_grad = True
                    self.conv2.weight.requires_grad = True
                    out = F.relu(self.bn1(x))
                    shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
                    out = self.conv1(out)
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += shortcut
                else:
                    print("!!!!!!!!!!!!!!!!!!")
                    all_block_count = all_block_count - 1
                    self.conv1.weight.requires_grad = False
                    self.conv2.weight.requires_grad = False

                    out = F.relu(self.bn1(x))
                    out = self.shortcut(out) if hasattr(self, 'shortcut') else x

            return out
        else:
            out = F.relu(self.bn1(x))
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.conv2(F.relu(self.bn2(out)))
            out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet_sto(nn.Module):
    def __init__(self, block,prob_0_L, num_blocks, num_classes=200):
        super(PreActResNet_sto, self).__init__()
        self.in_planes = 64
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta / (sum(num_blocks) - 1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion*4, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.prob_now,self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            self.prob_now = self.prob_now - self.prob_step
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# def PreActResNet18():
#     return PreActResNet(PreActBlock, [2,2,2,2])
# def PreActResNet18():
#     return PreActResNet_sto(PreActBlock, prob_0_L=[1,0.5], num_blocks = [2,2,2,2])











def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--prob_0', default=1.0, type=float)
    parser.add_argument('--prob_1', default=0.4, type=float)
    parser.add_argument('--model', default='PreActResNest18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous','normal'],
        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--factor', default=0.5, type=float, help='Label Smoothing')
    parser.add_argument('--rate', default=0.04, type=float, help='Label Smoothing')
    return parser.parse_args()
from torch.nn import functional as F


def _label_smoothing(label, factor):
    one_hot = np.eye(200)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(200 - 1))

    return result


from torch.autograd import Variable


def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss

def main():
    args = get_args()
    output_path = os.path.join(args.out_dir, 'epochs_' + str(args.epochs))
    output_path = os.path.join(output_path, 'lr_max_' + str(args.lr_max))
    output_path=os.path.join(output_path,'model_'+args.model)
    output_path = os.path.join(output_path, 'lr_schedule_' + str(args.lr_schedule))
    output_path=os.path.join(output_path, 'alpha_' + str(args.alpha))
    output_path = os.path.join(output_path, 'prob_0_' + str(args.prob_0))
    output_path = os.path.join(output_path, 'prob_1_' + str(args.prob_1))
    output_path = os.path.join(output_path, 'rate_' + str(args.rate))
    output_path = os.path.join(output_path, 'factor_' + str(args.factor))


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

    train_loader, test_loader = New_ImageNet_get_loaders_64(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std


    # model = PreActResNet18().cuda()
    # model.train()
    print('==> Building model..')
    logger.info('==> Building model..')

    model_sto = PreActResNet_sto(PreActBlock, prob_0_L=[args.prob_0, args.prob_1], num_blocks = [2,2,2,2])

    model_sto=model_sto.cuda()
    model_sto.train()
    opt = torch.optim.SGD(model_sto.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    #criterion = nn.CrossEntropyLoss()



    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100/110, lr_steps * 105/ 110], gamma=0.1)

    # Training
    prev_robust_acc = 0.
    #start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    best_result = 0
    epoch_clean_list = []
    epoch_pgd_list = []

    final_loss = []
    final_acc = []
    best_acc = 0
    cur_acc = 0
    cur_prob = args.prob_1
    all_time=0
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        if epoch > 0:

            if best_acc >= cur_acc:
                cur_prob = cur_prob +args.rate
                if cur_prob >= 0.9:
                    cur_prob = 0.9
                model_sto = PreActResNet_sto(PreActBlock, prob_0_L=[args.prob_0,cur_prob], num_blocks = [2,2,2,2])

                model_sto = model_sto.cuda()
                model_sto.load_state_dict(model_test.state_dict())
                model_sto.train()
                best_acc = cur_acc
            else:
                best_acc = cur_acc
                model_sto = PreActResNet_sto(PreActBlock, prob_0_L=[args.prob_0,cur_prob], num_blocks = [2,2,2,2])

                model_sto = model_sto.cuda()
                model_sto.load_state_dict(model_test.state_dict())
                model_sto.train()
        if epoch < 100:
            opt = torch.optim.SGD(model_sto.parameters(), lr=0.1, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps * 100 / 110,
                                                                              lr_steps * 105 / 110],
                                                             gamma=1)
        elif 100 <= epoch < 105:
            opt = torch.optim.SGD(model_sto.parameters(), lr=0.01, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                             milestones=[lr_steps * 100 / 110,
                                                                         lr_steps * 105 / 110],
                                                             gamma=1)
        elif 105 <= epoch:
            opt = torch.optim.SGD(model_sto.parameters(), lr=0.001, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                             milestones=[lr_steps * 100 / 110,
                                                                         lr_steps * 105 / 110],
                                                             gamma=1)

        model_sto.train()
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)
            label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).cuda()).float()
            if args.delta_init != 'previous':
                delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for j in range(len(epsilon)):
                    delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            global m_sample_list
            global all_block_count
            delta.requires_grad = True
            output = model_sto(X + delta[:X.size(0)])
            loss = LabelSmoothLoss(output, (label_smoothing).float())
            print(m_sample_list)
            print(all_block_count)
            #with amp.scale_loss(loss, opt) as scaled_loss:
            loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
            delta = delta.detach()
            output = model_sto(X + delta[:X.size(0)])
            loss = LabelSmoothLoss(output, (label_smoothing).float())
            opt.zero_grad()
            #with amp.scale_loss(loss, opt) as scaled_loss:
            loss.backward()
            print(m_sample_list)
            print(all_block_count)
            m_sample_list = []
            all_block_count = 0
            torch.nn.utils.clip_grad_norm_(model_sto.parameters(), 10)

            print(loss)
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        final_loss.append(train_loss / train_n)
        final_acc.append(train_acc / train_n)
        cur_acc=train_acc/train_n
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        all_time=all_time+epoch_time - start_epoch_time
        logger.info('==> Building model..')

        model_test = PreActResNet_sto(PreActBlock, prob_0_L=[args.prob_0,cur_prob], num_blocks = [2,2,2,2]).cuda()



        model_test.load_state_dict(model_sto.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_sto, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_sto)
        epoch_clean_list.append(test_acc)
        epoch_pgd_list.append(pgd_acc)
        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)
        if best_result<=pgd_acc:
            best_result=pgd_acc
            torch.save(model_sto.state_dict(), os.path.join(output_path, 'best_model.pth'))


    torch.save(model_sto.state_dict(), os.path.join(output_path, 'final_model.pth'))
    logger.info(epoch_clean_list)
    logger.info(epoch_pgd_list)
    print(epoch_clean_list)
    print(epoch_pgd_list)
    logger.info("Training")
    # logger.info(final_loss)
    # logger.info(final_acc)
    # print('final_loss', final_loss)
    # print('final_acc', final_acc)
    print('all_time',all_time)

if __name__ == "__main__":
    main()
