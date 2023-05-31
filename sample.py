import matplotlib.pyplot as plt
import torch

from MLP import MLP
from LatentDataset import LatentDataSet
from torch.utils.data import DataLoader
import os
from ddpm import *
from ddpm_config import *
import numpy as np
import time

Cuda = False
device = torch.device("cuda" if Cuda else "cpu")

# 准备隐变量数据集
if latent_size == 32:
    dir = r'data'
    latent_dir = 'sample_latent/'
else:
    dir = r'data2'
    latent_dir = 'sample_latent_test'
files = os.listdir(dir)
lines = []
for item in files:
    line = dir + '\\' + item
    lines.append(line)

dataset = LatentDataSet(lines)

# 开始训练模型，打印loss及中间重构效果
seed = 1234

print('Training model...')
batch_size = 1
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MLP(latent_size, device=device)  # 输出维度是2，输入是x和step
load_weight = True
if load_weight:
    # path = r'result/weight32/weight-39998.0000_test_loss-0.0425.pth'
    path = r'result/weight/weight-9999.0000_test_loss-0.0267.pth'
    weight = torch.load(path)
    model.load_state_dict(weight)
    print('load weight')


# 采样生成
shape = [1, 1, latent_size]

x_set = []
latent_set = []
x_cat = None
latent_cat = None

for d in range(50, 5000, 200):
    depth = torch.tensor([d], dtype=torch.long)
    print('sample depth %d...' % d)
    s_t = time.time()
    x_seq = p_sample_loop(model, shape, num_steps, depth, betas, one_minus_alphas_bar_sqrt, device)
    e_t = time.time()
    print(e_t-s_t)
    for i in range(0, 11):
        filename = latent_dir + str(i+1) + '/' + str(int(depth.item())) + '.latent'
        torch.save(x_seq[i*100], filename)

    for i in range(1, 11):
        if x_cat is not None:
            x_cat = torch.cat((x_cat, x_seq[100 * i]), 1)
        else:
            x_cat = x_seq[100 * i]

    if latent_cat is not None:
        latent_cat = torch.cat((latent_cat, x_seq[-1]), 1)
    else:
        latent_cat = x_seq[100]

    x_cat_np = x_cat.reshape([10, latent_size]).cpu().detach()
    x_cat = None
    x_cat_np = np.array(x_cat_np)
    x_set.append(x_cat_np)
    # plt.figure(1)
    # plt.imshow(x_cat_np)
    # plt.show()
#
# latent_cat_np = np.array(latent_cat.reshape([25, latent_size]).cpu().detach())
# plt.figure(1)
# plt.imshow(latent_cat_np)
# plt.show()

print('finish')








