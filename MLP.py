import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, length, n_steps=1000, depth_dim=5000, num_units=256, device='cuda'):
        super(MLP, self).__init__()
        self.device = device
        self.linears = nn.ModuleList(
            [
                nn.Linear(length, num_units, bias=False, device=device),
                nn.SiLU(),
                nn.Linear(num_units, num_units, bias=False, device=device),
                nn.SiLU(),
                nn.Linear(num_units, num_units, bias=False, device=device),
                nn.SiLU(),
                # nn.Linear(num_units, num_units, bias=False, device=device),
                # nn.SiLU(),
                nn.Linear(num_units, length, bias=False, device=device),
            ]
        )

        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units, device=device),
                nn.Embedding(n_steps, num_units, device=device),
                nn.Embedding(n_steps, num_units, device=device),
                # nn.Embedding(n_steps, num_units),
            ]
        )

        self.depth_embeddings = nn.ModuleList(
            [
                nn.Embedding(depth_dim, num_units, device=device),
                nn.Embedding(depth_dim, num_units, device=device),
                nn.Embedding(depth_dim, num_units, device=device),
                # nn.Embedding(depth_dim, num_units),
            ]
        )

    def forward(self, x, t, depth):
        x = x.to(self.device)
        t = t.to(self.device)
        depth = depth.to(self.device)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            d_embedding = self.depth_embeddings[idx](depth)
            t_embedding = t_embedding.to(self.device)
            d_embedding = d_embedding.to(self.device)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x += d_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)
        return x


if __name__ == '__main__':
    import os
    from LatentDataset import LatentDataSet
    from torch.utils.data import DataLoader

    model = MLP(32)

    dir = r'../old_tl/latent'
    files = os.listdir(dir)
    lines = []
    for item in files:
        line = dir + '\\' + item
        lines.append(line)

    dataset = LatentDataSet(lines)
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for batch_idx, (depth, latent) in enumerate(dataloader):
        t = torch.LongTensor([1])
        t = t.reshape([batch_size, 1])
        y = model(latent, t, depth)
        print(y)
        break














