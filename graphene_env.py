import os
import csv
import math
import copy
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, resnet50

# from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atom, Atoms
from ase import *
from ase.visualize import view
from ase.io import write
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdchem
from rdkit.Chem.AtomPairs import Pairs

from gym.utils import seeding
from gym import spaces

from cnn import MLP
import xgboost as xgb
from utils import coord2contour_img, coord2feature, cal_distance, nearest_atom


# build the prediction model
class PredictModel(nn.Module):
    def __init__(self, model_dir, device):
        super(PredictModel, self).__init__()
        self.resnet_flux = resnet50(pretrained=False)
        # self.resnet_flux = resnet18(pretrained=False)
        self.mlp_flux = MLP(in_size=1000)
        self.resnet_rej = resnet50(pretrained=False)
        # self.resnet_rej = resnet18(pretrained=False)
        self.mlp_rej = MLP(in_size=1000)

        self.resnet_flux.load_state_dict(torch.load(
            os.path.join(model_dir, 'resnet_smoothl1_all2_flux.ckpt'), map_location=device
        ))
        self.mlp_flux.load_state_dict(torch.load(
            os.path.join(model_dir, 'mlp_smoothl1_all2_flux.ckpt'), map_location=device
        ))
        self.resnet_rej.load_state_dict(torch.load(
            os.path.join(model_dir, 'resnet_smoothl1_all2_rej.ckpt'), map_location=device
        ))
        self.mlp_rej.load_state_dict(torch.load(
            os.path.join(model_dir, 'mlp_smoothl1_all2_rej.ckpt'), map_location=device
        ))

        self.device = device
        self.resnet_flux = self.resnet_flux.to(device).eval()
        self.mlp_flux = self.mlp_flux.to(device).eval()
        self.resnet_rej = self.resnet_rej.to(device).eval()
        self.mlp_rej = self.mlp_rej.to(device).eval()

        with open(os.path.join(model_dir, 'mean_std_all.pickle'), 'rb') as handle:
            self.mean_std = pickle.load(handle)
            print(self.mean_std)

        self.embed = None


    def forward(self, x):
        if len(x.shape) < 4:
            x = np.expand_dims(x, axis=0)
        batch_size = x.shape[0]
        x = torch.from_numpy(x).type(torch.FloatTensor)
        out = self.resnet_flux(x)
        flux, flux_embed = self.mlp_flux(out)
        out = self.resnet_rej(x)
        rej, rej_embed = self.mlp_rej(out)

        out = torch.stack((flux, rej), dim=-1)
        embed = torch.cat((flux_embed, rej_embed), dim=1)

        if self.device is 'cpu':
            out = out.detach().numpy()
            embed = embed.detach().numpy()
        else:
            out = out.detach().cpu().numpy()
            embed = embed.detach().cpu().numpy()
        
        out[...,0] = out[...,0] * self.mean_std['flux_std'] + self.mean_std['flux_mean']
        out[...,1] = out[...,1] * self.mean_std['rej_std'] + self.mean_std['rej_mean']

        if batch_size == 1:
            out = np.squeeze(out)
            embed = np.squeeze(embed)

        self.embed = embed

        return out


class Graphene():
    def __init__(self, fn='4040.pdb', model_dir='./models', device='cpu'):
        # read graphene structure from .pdb file
        self.FILENAME = fn
        self.Z = '96.500'
        self.pdbStart = "CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1\n"
        # self.pdbContent = "ATOM      ind  C   GRA X   ind       xcoord   ycoord  96.500  0.00  0.00      SHT  C\n"
        self.pdbContent = "ATOM    ind  C   GRA X ind      xcoord  ycoord  96.500  0.00  0.00      SHT  C\n"
        self.pdbEnd = "END"
        fp = open(self.FILENAME, "r")
        contents = fp.readlines()[1:-1]
        self.origin_atom_coord = np.array([list(map(float,x.split()[6:8])) for x in contents])
        fp.close()

        mol = Chem.MolFromPDBFile(fn)
        self.mol = Chem.RWMol(mol)
        
        # get the index of atoms on the graphene edge 
        edge_idx = []
        edge_idx.extend(list(np.argwhere(self.origin_atom_coord[:,0] < 0.5).squeeze()))
        edge_idx.extend(list(np.argwhere(self.origin_atom_coord[:,0] > 40).squeeze()))
        edge_idx.extend(list(np.argwhere(self.origin_atom_coord[:,1] < 0.5).squeeze()))
        edge_idx.extend(list(np.argwhere(self.origin_atom_coord[:,1] > 41).squeeze()))
        self.edge_idx = set(edge_idx)

        self.norm_atom_coord = np.copy(self.origin_atom_coord)
        self.norm_atom_coord /= np.max(self.norm_atom_coord)
        self.atom_coord = np.copy(self.norm_atom_coord)

        self.idx = list(range(len(self.origin_atom_coord)))
        self.removable_idx = list(set(self.idx) - self.edge_idx)
        self.topo = self.get_topo(self.idx, self.origin_atom_coord)

        self.num_remove = 0             # number of atoms removed
        self.cur_remove_atom = None     # removed atom index at current step
        self.remove_idx = set()         # index of all removed atoms at current step
        self.remove_idx_seq = []        # sequence of all removed atoms in an episode
        self.num_neighbor = 15          # number of adjacent atoms to consider in each time step    

        self.prop_predictor = PredictModel(model_dir=model_dir, device=device)
        # self.xgbFlux = xgb.XGBRegressor()
        # self.xgbFlux.load_model('models/xgb_flux.model')
        # self.xgbRej = xgb.XGBRegressor()
        # self.xgbRej.load_model('models/xgb_rej.model')
        # self.xgbMeanStd = np.load('models/xgb_mean_std.npy')


    def get_topo(self, idx, coord):
        topo = {}
        for pos, ind in enumerate(idx):
            connection = []
            restID = idx[:pos] + idx[pos+1:]
            for i in restID:
                if cal_distance(coord[i], coord[ind]) < 1.43:
                    connection.append(i)
                if len(connection) > 2:
                    break
            topo[ind] = connection
        return topo

    def remove_from_topo(self, idx):
        if idx in self.topo.keys():
            connection = self.topo.pop(idx)
            for c in connection:
                try:
                    if idx in self.topo[c]:
                        self.topo[c].remove(idx)
                except:
                    print('Tried to remove %d, did not fint it' %idx)
        return

    def get_pore_edge(self):
        pore_edges = []
        for k, v in self.topo.items():
            if len(v) < 3 and k not in self.edge_idx:
                pore_edges.append(k)
        return pore_edges

    def get_candidate(self):
        candidate = self.get_pore_edge()
        if len(candidate) >= self.num_neighbor:
            return candidate[:self.num_neighbor]
        else:
            nn_indices = sorted(nearest_atom(self.origin_atom_coord, self.remove_idx, self.num_neighbor))
            idx = 0
            while len(candidate) < self.num_neighbor:
                if nn_indices[idx] not in candidate:
                    candidate.append(nn_indices[idx])
                idx += 1
            return candidate

    def reset(self, rm_idx=None):
        if rm_idx == None:
            if self.FILENAME == '2020.pdb':
                rm_idx = 95
            elif self.FILENAME == '4040.pdb':
                rm_idx = 307
        
        self.atom_coord = np.copy(self.norm_atom_coord)
        self.remove_idx = set()
        self.topo = self.get_topo(self.idx, self.origin_atom_coord)
        self.num_remove = 0
        self.remove_idx_seq = []
        self.num_remove_list = []
        self.cur_remove_atom = None

        self.atom_coord[rm_idx, :] = [0., 0.]
        self.remove_idx.add(rm_idx)
        self.num_remove += 1
        self.remove_from_topo(rm_idx)
        self.remove_idx_seq.append(list(self.remove_idx))
        self.num_remove_list.append(self.num_remove)

        # self.candidate = sorted(nearest_atom(self.origin_atom_coord, self.remove_idx, num_atoms=self.num_neighbor))

        self.candidate = self.get_candidate()


    def remove(self, rm_idx):
        if rm_idx == len(self.removable_idx):
            self.cur_remove_atom = None
            self.remove_idx_seq.append(list(self.remove_idx))
            self.num_remove_list.append(self.num_remove)
            return False
        
        rm_idx_true = self.removable_idx[rm_idx]
        if rm_idx_true in self.remove_idx or rm_idx_true == len(self.removable_idx):
            self.cur_remove_atom = None
            self.remove_idx_seq.append(list(self.remove_idx))
            self.num_remove_list.append(self.num_remove)
            return False
        else:
            self.cur_remove_atom = rm_idx_true
            self.atom_coord[rm_idx_true, :] = [0., 0.]
            self.remove_idx.add(rm_idx_true)
            self.num_remove += 1
            self.remove_from_topo(rm_idx_true)
            dangles = self.get_dangles()
            for atom in dangles:
                self.remove_from_topo(atom)
                self.atom_coord[atom, :] = [0., 0.]
                self.remove_idx.add(atom)
                self.num_remove += 1
            self.remove_idx_seq.append(list(self.remove_idx))
            self.num_remove_list.append(self.num_remove)
            return True


    def remove_neighbor(self, rm_idx):
        if rm_idx == self.num_neighbor:
            self.cur_remove_atom = None
            self.remove_idx_seq.append(list(self.remove_idx))
            self.num_remove_list.append(self.num_remove)
            return False
        
        # self.candidate = sorted(nearest_atom(self.origin_atom_coord, self.remove_idx, num_atoms=self.num_neighbor))
        self.candidate = self.get_candidate()
        rm_idx_true = self.candidate[rm_idx]

        if rm_idx_true in self.remove_idx or rm_idx_true in self.edge_idx:
            self.cur_remove_atom = None
            # self.remove_idx_seq.append(list(self.remove_idx))
            # self.num_remove_list.append(self.num_remove)
            rm_flag = False
            # return False
        
        else:
            self.cur_remove_atom = rm_idx_true
            self.atom_coord[rm_idx_true, :] = [0., 0.]
            self.remove_idx.add(rm_idx_true)
            self.num_remove += 1
            self.remove_from_topo(rm_idx_true)
            dangles = self.get_dangles()
            for atom in dangles:
                self.remove_from_topo(atom)
                self.atom_coord[atom, :] = [0., 0.]
                self.remove_idx.add(atom)
                self.num_remove += 1
            # self.remove_idx_seq.append(list(self.remove_idx))
            # self.num_remove_list.append(self.num_remove)
            rm_flag = True
            # return True

        self.remove_idx_seq.append(list(self.remove_idx))
        self.num_remove_list.append(self.num_remove)
        
        return rm_flag
        

    def get_dangles(self):
        remaining_atoms = set(self.topo.keys())
        # Construct a networkx graph using self.topo
        G = nx.from_dict_of_lists(self.topo)
        # Get all atoms that have connection to atom 1
        # Since it does not contain 1, atom 1 must be manually added
        non_dangling = nx.algorithms.descendants(G, 1)
        non_dangling.add(1)
        return remaining_atoms - non_dangling


    def get_pred_flux_rej(self, predictor='cnn'):
        # img_dir = 'process_img'
        # img_fn = '0.png'
        # img_path = os.path.join(img_dir, img_fn)

        # remain_indices = list(set(self.removable_idx) - self.remove_idx)
        remain_indices = list(set(list(range(len(self.origin_atom_coord)))) - self.remove_idx)
        remain_coords = self.origin_atom_coord[remain_indices, :]

        if predictor == 'cnn':
            img = coord2contour_img(remain_coords, img_fn=None)
        
            transform = transforms.Compose([
                transforms.CenterCrop((380, 380)),
                transforms.Resize((224, 224))
            ])

            # img = Image.fromarray(img)
            img = img.convert("RGB")
            img = transform(img)
            # img = np.array(img)[:,:,:3]
            img = np.rollaxis(np.array(img), 2, 0) / 255.0

            pred = self.prop_predictor(img)
            # print(pred)
            flux, rej = pred[0], pred[1]

            # flux = pred[0] * self.mean_std['flux_std'] + self.mean_std['flux_mean']
            # rej = pred[1] * self.mean_std['rej_std'] + self.mean_std['rej_mean']
        
        else:
            feature = coord2feature(remain_coords)
            # Predict
            sFlux = self.xgbFlux.predict(feature)[0]
            sRej = self.xgbRej.predict(feature)[0]
            
            flux = sFlux * self.xgbMeanStd[0, 1] + self.xgbMeanStd[0, 0]
            rej = sRej * self.xgbMeanStd[1, 1] + self.xgbMeanStd[1, 0]

        # print(flux, rej)

        return flux, rej


    def get_pseudo_flux_rej(self):
        if self.num_remove < 10:
            return 0, 0
        else:
            # return 0.8 * self.num_remove, max([1 - 0.005 * self.num_remove**2, 0])
            return 0.8 * self.num_remove, 0

    
    def get_coord(self):
        return self.atom_coord.flatten()


    def get_fingerprint(self):
        n_features = 1024
        cutoff = 5

        reverse_rm_indices = sorted(self.remove_idx, reverse=True)

        temp_mol = copy.deepcopy(self.mol)
        for idx in reverse_rm_indices:
            # print(idx)
            temp_mol.RemoveAtom(int(idx))
        # print()

        # Step 1: Convert Mol object to Morgan Fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(temp_mol, cutoff, useFeatures=True, nBits=n_features)
        # Step 2: Convert Morgan Fingerprints to numpy array
        array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, array)
        
        return array


    def get_removed_encode(self):
        rm_embed = np.zeros(len(self.origin_atom_coord))
        if len(self.remove_idx) > 0:
            rm_embed[list(self.remove_idx)] = 1.0
        return rm_embed


    def get_candidate_encode(self):
        # cand_embed = np.zeros(len(self.origin_atom_coord))
        # cand_embed[list(self.candidate)] = 1.0
        # return cand_embed
        return np.array(self.candidate)

    
    def plot(self, save_path=None):
        rm_idx = self.remove_idx_seq[-1]
        x_remain = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i not in rm_idx]
        y_remain = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i not in rm_idx]
        
        fig = plt.figure(figsize=[5,5], dpi=600)
        plt.scatter(x_remain, y_remain, c='black')
        plt.axis('equal')
        plt.axis('off')

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    def plot_cnn_input(self):
        remain_indices = list(set(list(range(len(self.origin_atom_coord)))) - self.remove_idx)
        coords = self.origin_atom_coord[remain_indices, :]

        # img = coord2contour_img(remain_coords, img_fn=None)
        # img.show()

        fig, ax = plt.subplots()
        DPI = fig.get_dpi()
        fig.set_size_inches(500.0/float(DPI), 500.0/float(DPI))

        ax.scatter(coords[:,0], coords[:,1], c='blue', s=400)
        ax.scatter(coords[:,0], coords[:,1], c='green', s=200)
        ax.scatter(coords[:,0], coords[:,1], c='yellow', s=60)
        ax.scatter(coords[:,0], coords[:,1], c='red', s=10)
        ax.axis('off')
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # fig.canvas.draw()
        # X = np.array(fig.canvas.renderer.buffer_rgba())
        plt.show()
        plt.close()


    def plot_candidate(self, save_path=None):
        # x_remain = [self.atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i not in self.remove_idx]
        # y_remain = [self.atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i not in self.remove_idx]
        # x_remove = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i in self.candidate]
        # y_remove = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i in self.candidate]
        x_remain, y_remain = [], []
        x_cand, y_cand = [], []
        remain_idx = list(range(len(self.origin_atom_coord)))
        for i in remain_idx:
            if i in self.candidate:
                x_cand.append(self.origin_atom_coord[i][0])
                y_cand.append(self.origin_atom_coord[i][1])
            elif i not in self.remove_idx:
                x_remain.append(self.origin_atom_coord[i][0])
                y_remain.append(self.origin_atom_coord[i][1])

        plt.figure()
        plt.scatter(x_remain, y_remain, c='black')
        plt.scatter(x_cand, y_cand, c='red')
        plt.axis('equal')
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
    

    def plot_seq(self, save_path=None):
        n_rows = 8
        n_cols = len(self.remove_idx_seq) // n_rows
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols, n_rows), dpi=600)
        counter = 0
        for row in ax:
            for col in row:
                rm_idx = self.remove_idx_seq[counter]
                x_remain = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i not in rm_idx]
                y_remain = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i not in rm_idx]
                # x_remove = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i in rm_idx]
                # y_remove = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i in rm_idx]
                col.scatter(x_remain, y_remain, c='black', s=0.25)
                # col.scatter(x_remove, y_remove, c='red')                
                col.axis('equal')
                col.axis('off')
                counter += 1
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        # save_dir = save_path[:-4]
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # for i, remove_idx in enumerate(self.remove_idx_seq):
        #     atoms = Atoms()
        #     for j in range(len(self.origin_atom_coord)):
        #         if not (j in remove_idx):
        #             carbon = Atom('C', (self.origin_atom_coord[j][0], self.origin_atom_coord[j][1], 96.5))
        #             atoms.append(carbon)
        #     cell = [[40.525+1.228,0,0],[0,41.122+1.418,0],[0,0,200]]
        #     atoms.set_cell(cell)
        #     img_path = os.path.join(save_dir, str(i) + '.png')
        #     write(img_path, atoms)


    def plot_neighbor(self, save_path=None):
        # candidate = sorted(nearest_atom(self.origin_atom_coord, self.remove_idx, num_atoms=self.num_neighbor))
        x_remain = [self.atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i not in self.remove_idx]
        y_remain = [self.atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i not in self.remove_idx]
        x_remove = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i in self.remove_idx]
        y_remove = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i in self.remove_idx]
        # x_neighbor = [self.origin_atom_coord[i][0] for i in range(self.atom_coord.shape[0]) if i in candidate]
        # y_neighbor = [self.origin_atom_coord[i][1] for i in range(self.atom_coord.shape[0]) if i in candidate]
        plt.figure()
        plt.scatter(x_remain, y_remain, c='black')
        plt.scatter(x_remove, y_remove, c='red')
        # plt.scatter(x_neighbor, y_neighbor, c='blue')
        plt.axis('equal')
        if save_path is not None:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


    def write2pdb(self, pdbname='graphene.pdb'):
        coords = []
        for i in range(len(self.origin_atom_coord)):
            if i not in self.remove_idx:
                coords.append(self.origin_atom_coord[i, :])
        coords = np.array(coords)
        
        pdb_file = open(pdbname, "w")
        pdb_file.write(self.pdbStart)
        for i, coord in enumerate(coords):
            # to_write = self.pdbContent.replace('ind',
            #                             str(i+1)).replace('xcoord',
            #                             str(coord[0])).replace('ycoord',str(coord[1]))
            x = coord[0]
            y = coord[1]
            to_write = self.pdbContent.replace('ind',
                                    str(i+1).rjust(3)).replace('xcoord', 
                                    "{:.3f}".format(x).rjust(6)).replace('ycoord',
                                    "{:.3f}".format(y).rjust(6))
            pdb_file.write(to_write)
        pdb_file.write(self.pdbEnd)
        pdb_file.close()


class GrapheneEnv():
    def __init__(self, max_timesteps, fn='4040.pdb'):
        self.graphene = Graphene(fn)
        self.max_timesteps = max_timesteps
        self.terminal = False
        self.steps = 0
        self.state = None
        self.reward = 0
        self.rew_list = []
        self.flux_list, self.rej_list = [], []
        self.flux, self.rej = 0.0, 0.0
        self.ep = 0

        self.reset()
        
        state = self.get_state()
        max_coord = np.max(state)
        high = np.repeat(max_coord*2, len(state)).astype(np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # self.action_space = spaces.Discrete(len(self.graphene.removable_idx)+1)
        self.action_space = spaces.Discrete(self.graphene.num_neighbor+1)


    def get_state(self):
        coord = self.graphene.get_coord()
        fp = self.graphene.get_fingerprint()
        # rm_encode = self.graphene.get_removed_encode()
        cand_encode = self.graphene.get_candidate_encode()
        img_embed = self.graphene.prop_predictor.embed
        state = np.concatenate([coord, fp, cand_encode, img_embed])
        # return self.graphene.atom_coord.flatten()
        return state


    def get_flux_rej(self):
        return self.graphene.get_pred_flux_rej()

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.graphene.reset()
        self.terminal = False
        self.steps = 0
        flux, rej = self.get_flux_rej()
        self.flux = max(0.0, flux)
        self.rej = min(1.0, rej)
        self.state = self.get_state()
        self.reward = 0.0
        self.rew_list = []
        self.flux_list, self.rej_list = [self.flux], [self.rej]
        return self.state


    def step(self, action):
        self.steps += 1
        # rm = self.graphene.remove(rm_idx=action)
        rm = self.graphene.remove_neighbor(rm_idx=action)
        rew = 0.0
        
        if rm is True:
            curr_flux, curr_rej = self.get_flux_rej()

            curr_flux = max(0.0, curr_flux)
            curr_rej = min(1.0, curr_rej)

            # rew += (curr_flux - self.flux_list[0]) * 0.02
            # rew -= 10 * (curr_rej - self.rej_list[0])**2

            self.flux, self.rej = curr_flux, curr_rej
            # self.flux_list.append(curr_flux)
            # self.rej_list.append(curr_rej)

            # rew += 0.01
            rew += 0.05
        
        # rew += self.flux * 0.05
        # rew += math.log((max(0.855, self.rej) - 0.85)/0.15)

        rew += self.flux * 0.01
        # rew -= math.exp((0.85-self.rej)/0.1)

        # def rejReward(rej, A=-50, K=0, B=14, Q=100, v=0.01, C=1):
        #     return A + (K-A)/((C+Q*np.exp(-B*rej))**(1/v))
        def logistic(rej, A=-10, K=0, B=13, Q=100, v=0.01, C=1):
            return A + (K-A)/((C+Q*np.exp(-B*rej))**(1/v))
        # rew += logistic(self.rej)
        rew += logistic(self.rej,-15,0,13,100,0.01,1) - logistic(1,-15,0,13,100,0.01,1)

        self.flux_list.append(self.flux)
        self.rej_list.append(self.rej)
        
        self.state = self.get_state()

        if self.steps >= self.max_timesteps:
            self.terminal = True
            self.ep += 1
            print("number of atoms removed: {}, number of steps: {}".format(self.graphene.num_remove, self.steps))

        self.reward += rew
        self.rew_list.append(self.reward)
        return self.state, rew, self.terminal, {}


    def close(self):
        self.reset()


    def visualize(self, save_path=None):
        self.graphene.plot(save_path)


    def visualize_seq(self, save_path=None):
        self.graphene.plot_seq(save_path)
        # print("figure saved as", save_path)


    def plot_flux(self, save_path=None):
        plt.figure()
        plt.plot(np.arange(len(self.flux_list)), self.flux_list)
        plt.xlabel('timestep')
        plt.ylabel('water flux')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            # print('figure saved as {}'.format(save_path))
            plt.close()
        else:
            plt.show()


    def plot_ion_rej(self, save_path=None):
        plt.figure()
        plt.plot(np.arange(len(self.rej_list)), self.rej_list)
        plt.xlabel('timestep')
        plt.ylabel('ion rejection')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            # print('figure saved as {}'.format(save_path))

            plt.close()
        else:
            plt.show()


    def plot_remove_number(self, save_path=None):
        plt.figure()
        plt.plot(np.arange(len(self.graphene.num_remove_list)), self.graphene.num_remove_list)
        plt.xlabel('timestep')
        plt.ylabel('number of removed atoms')
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            # print('figure saved as {}'.format(save_path))
            plt.close()
        else:
            plt.show()
