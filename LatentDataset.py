import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ddpm_config import *


class LatentDataSet(Dataset):
    def __init__(self, lines):
        super(LatentDataSet, self).__init__()
        self.lines = lines
        self.length = len(lines)

    def __getitem__(self, index):
        line = self.lines[index].strip()
        latent = torch.load(line)
        latent = latent.reshape([1, latent_size])
        s = line.split('\\')
        depth = int(s[-1].split('.')[0])
        depth = torch.tensor(depth, dtype=torch.long)
        return depth, latent

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import os
    dir = r'../old_tl/latent'
    files = os.listdir(dir)
    lines = []
    for item in files:
        line = dir + '\\' + item
        lines.append(line)

    dataset = LatentDataSet(lines)
    dataloader = DataLoader(dataset, batch_size=2)
    for batch_idx, (depth, latent) in enumerate(dataloader):
        print(depth)
        print(latent.shape)


