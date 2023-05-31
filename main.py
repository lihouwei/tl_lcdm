import matplotlib.pyplot as plt
import torch

from MLP import MLP
from LatentDataset import LatentDataSet
from torch.utils.data import DataLoader
import os
from ddpm import *
from ddpm_config import *


def save_list(arr, filename, mode):
    fd = open(filename, mode)
    for val in arr:
        val = str(val) + '\n'
        fd.writelines(val)
    fd.close()


# 准备隐变量数据集
if latent_size == 32:
    dir = r'../old_tl/latent'
else:
    dir = r'../old_tl/latent64'
files = os.listdir(dir)
lines = []
for item in files:
    line = dir + '\\' + item
    lines.append(line)

dataset = LatentDataSet(lines)

# for batch_idx, (depth, latent) in enumerate(dataloader):
#     print(depth)
#     print(latent.shape)


# 开始训练模型，打印loss及中间重构效果
seed = 5678

print('Training model...')
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_epoch = 60000
# latent_size = 64
num_units = 256
model = MLP(latent_size, n_steps=num_steps, num_units=num_units, device='cuda')  # 输出维度是2，输入是x和step
load_weight = False
if load_weight:
    path = r'result/weight_64/weight-60000.0000_test_loss-0.0722.pth'
    weight = torch.load(path)
    model.load_state_dict(weight)
    print('load weight')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_loss = []
loss_arr = []

for epoch in range(num_epoch):
    for idx, (depth, latent) in enumerate(dataloader):
        latent = latent.reshape([-1, latent_size])
        loss = diffusion_loss_fn(model, latent, depth, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        epoch_loss.append(loss.item())
    # print('epoch %d, loss = %.4f' % (epoch, sum(epoch_loss) / len(epoch_loss)))
    loss_arr.append(sum(epoch_loss) / len(epoch_loss))
    if epoch % 500 == 0 or (num_epoch - epoch) < 5:
        print('epoch %d, loss = %.4f' % (epoch, sum(epoch_loss) / len(epoch_loss)))
        # weight = './result/weight64-2/weight-%.4f_test_loss-%.4f.pth' % (epoch + 1, sum(epoch_loss)/len(epoch_loss))
        # epoch_loss = []
        # torch.save(model.state_dict(), weight)

file = 'result/loss/loss32' + '.txt'
save_list(loss_arr, file, 'w')









