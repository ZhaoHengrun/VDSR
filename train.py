from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_valid_set

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # use the chosen gpu
torch.cuda.set_device(1)  # use the chosen gpu

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', default=4, type=int, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=8000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-3, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

last_psnr = 0
current_psnr = 0
psnr_increase = 0
psnr_increase_last = 0
psnr_increase_last_2 = 0
psnr_increase_last_3 = 0
psnr_increase_last_4 = 0
best_psnr = 0
last_best_epoch = 0
last_adjust_epoch = 0
psnr_increase_avg = 0

psnr_stable_count = 0
lr_adjust_flag = 0
stop_flag = False
save_flag = False
lr = opt.lr

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_valid_set()
training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)  # num_workers=opt.threads
testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False)  # num_workers

print('===> Building model')
model = Net().to(device)
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        # print(
        #     "===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch [{}] Complete: Avg.Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def valid():
    avg_psnr = 0
    global current_psnr
    global last_psnr
    global psnr_increase
    global psnr_increase_last
    global psnr_increase_last_2
    global psnr_increase_last_3
    global psnr_increase_last_4
    global psnr_increase_avg
    global best_psnr
    global save_flag

    psnr_increase_last_4 = psnr_increase_last_3
    psnr_increase_last_3 = psnr_increase_last_2
    psnr_increase_last_2 = psnr_increase_last
    psnr_increase_last = psnr_increase
    with torch.no_grad():
        last_psnr = current_psnr
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    avg_psnr = avg_psnr / len(testing_data_loader)
    current_psnr = avg_psnr
    if current_psnr > best_psnr:
        best_psnr = current_psnr
        save_flag = True
    else:
        save_flag = False
    psnr_increase = current_psnr - last_psnr
    psnr_increase_avg = (
                                psnr_increase + psnr_increase_last + psnr_increase_last_2 +
                                psnr_increase_last_3 + psnr_increase_last_4) / 5
    print(
        "===> Avg.PSNR: [{:.4f}], psnr_increase: [{:.4f}], "
        "psnr_increase_avg: [{:.4f}]".format(avg_psnr, psnr_increase,
                                             psnr_increase_avg))


def checkpoint(epoch):
    global last_best_epoch
    if epoch % 10 == 0:
        model_out_path = "checkpoints/epoch_{}_psnr_{:.2f}.pth".format(epoch, current_psnr)
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        torch.save(model, "checkpoints/current_epoch.pth")
    if save_flag is True:
        print('|||||||||||||||||||||||||||||||||  here, best psnr is {:.4f}  |||||||||||||||||||||||||||||||||'.format(
            current_psnr))
        torch.save(model, "checkpoints/best_psnr{:.4f}_epoch{}.pth".format(current_psnr, epoch))
        last_best_epoch = epoch


def adjust_lr():
    global psnr_increase
    global psnr_stable_count
    global lr
    global stop_flag
    global last_best_epoch
    global last_adjust_epoch

    if lr < 1e-6:
        stop_flag = True

    # if 0.001 > psnr_increase:  # if 0.001 > psnr_increase:
    #     psnr_stable_count += 1
    #     print('//////////////////////////psnr_increase is slow enough! [{}]///////////////////////////'.format(
    #         psnr_stable_count))
    # else:
    #     psnr_stable_count = 0

    # if psnr_stable_count == 5:
    # if psnr_increase_avg < 0.001:
    if epoch - last_best_epoch > 40 and epoch - last_adjust_epoch > 40:
        lr = lr / 2
        psnr_stable_count = 0
        print('================================lr decrease to [{:.8f}]================================'.format(lr))
        last_adjust_epoch = epoch
    print("===> Current lr: [{:.8f}], last_best_epoch is [{}], last_adjust_epoch is [{}]".format(lr, last_best_epoch,
                                                                                                 last_adjust_epoch))


for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    valid()
    checkpoint(epoch)
    adjust_lr()
    if stop_flag is True:
        break
else:
    print('Done!')
