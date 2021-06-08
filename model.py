import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)


class StateEmbed(nn.Module):
    def __init__(self, nb_states, nb_coords=1360, nb_fingerprint=1024, nb_candidate=20, nb_imgfeat=128):
        super(StateEmbed, self).__init__()

        self.nb_coords = nb_coords
        self.nb_fingerprint = nb_fingerprint
        self.nb_candidate = nb_candidate
        self.nb_imgfeat = nb_imgfeat

        self.coord_embed = nn.Sequential(
            nn.Linear(nb_coords, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        self.fp_embed = nn.Sequential(
            nn.Linear(nb_fingerprint, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        self.candidate_embed1 = nn.Embedding(608, 16)
        self.candidate_embed2 = nn.Sequential(
            nn.Linear(16*nb_candidate, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        self.img_embed = nn.Sequential(
            nn.Linear(nb_imgfeat, 64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        emb1 = self.coord_embed(x[..., :self.nb_coords])
        emb2 = self.fp_embed(x[..., self.nb_coords:self.nb_coords+self.nb_fingerprint])
        cand_embed = self.candidate_embed1(x[..., -self.nb_candidate-self.nb_imgfeat:-self.nb_imgfeat].type(torch.LongTensor))
        cand_embed = cand_embed.view(batch_size, -1)
        emb3 = self.candidate_embed2(cand_embed)
        emb4 = self.img_embed(x[..., -self.nb_imgfeat:])
        return torch.cat((emb1, emb2, emb3, emb4), dim=len(x.shape)-1)


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            StateEmbed(in_dim),
            nn.Linear(64*4, 64), 
            # nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.model(x)
