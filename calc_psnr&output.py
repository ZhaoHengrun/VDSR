from __future__ import print_function

import argparse

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor

from dataset import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default='datasets/test/test.png', help='input image to use')
parser.add_argument('--input_LR_path', type=str, default='datasets/test/set5/LR/', help='input path to use')
parser.add_argument('--input_HR_path', type=str, default='datasets/test/set5/HR/', help='input path to use')
parser.add_argument('--model', type=str, default='checkpoints/current_epoch.pth', help='model file to use')
parser.add_argument('--output_path', default='results/', type=str, help='where to save the output image')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


# def calc_psnr(img1, img2):
#     criterion = nn.MSELoss()
#     mse = criterion(img1, img2)
#     return 10 * log10(1 / mse.item())


loader = transforms.Compose([
    transforms.ToTensor()])

path = opt.input_LR_path
path_HR = opt.input_HR_path

crop_size = 256
scale = 4
# for i in range(image_nums):
image_nums = len([lists for lists in listdir(path) if is_image_file('{}/{}'.format(path, lists))])
print(image_nums)
psnr_avg = 0
psnr_avg_bicubic = 0
for i in listdir(path):
    if is_image_file(i):
        img_name = i.split('.')
        img_num = img_name[0]

        img_original = Image.open('{}{}'.format(path_HR, i))
        img_original_ybr = img_original.convert('YCbCr')
        img_original_y, _, _ = img_original_ybr.split()

        img_LR = Image.open('{}{}'.format(path, i))
        img_LR_ybr = img_LR.convert('YCbCr')
        y, cb, cr = img_LR_ybr.split()

        img_to_tensor = ToTensor()
        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

        model = torch.load(opt.model)
        if opt.cuda:
            model = model.cuda()
            input = input.cuda()

        out = model(input)

        out = out.cpu()
        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        psnr_val = calc_psnr(loader(out_img_y).unsqueeze(0), loader(img_original_y))
        psnr_avg += psnr_val
        print(psnr_val)

        psnr_val_bicubic = calc_psnr(loader(y).unsqueeze(0), loader(img_original_y))
        psnr_avg_bicubic += psnr_val_bicubic

        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        # out_img.save('{}{}_psnr_{:.2f}.png'.format(opt.output_path, img_num, psnr_val))
        out_img.save('{}output/{}.png'.format(opt.output_path, img_num))
        print('output image saved to ', opt.output_path)
        # img_LR.convert('RGB').save('{}{}_bicubic_{:.2f}.png'.format(opt.output_path, img_num, psnr_val_bicubic))
        img_original.save('{}gt/{}.png'.format(opt.output_path, img_num))

psnr_avg = psnr_avg / image_nums
psnr_avg_bicubic = psnr_avg_bicubic / image_nums
print('psnr_avg_bicubic', psnr_avg_bicubic)
print('psnr_avg', psnr_avg)
