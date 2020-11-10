# VDSR
一个VDSR的pytorch复现，在原版基础上瞎搞，没有彻底完善，仅仅是能跑。使用adam优化器，尝试了不同的学习率调整策略（但是没一个靠谱的），将网络输入的图像块大小改为256*256。 <br/>
参考自https://github.com/pytorch/examples/tree/master/super_resolution	<br/>
https://github.com/twtygqyy/pytorch-vdsr
## Requirements
Python 3.8<br/>
PyTorch 1.6.0<br/>
Numpy 1.19.2<br/>
Pillow 7.2.0<br/>
## Usage:
### Make datasets
使用make_datasets/目录下的make_dataset.m制作数据集，训练，验证，测试的图片分别保存在datasets/train，datasets/valid，datasets/test目录下	<br/>
### Train
运行`python main.py`进行训练	<br/>
模型保存在checkpoints/目录下	<br/>
### Test
运行`python calc_psnr&output.py`进行测试，生成的图片保存在results/目录下	<br/>
### Eval
`calc_psnr&output.py`中写的psnr计算公式可能有误差，也可使用make_datasets/目录下的`compute_psnr.m`计算psnr	<br/>
