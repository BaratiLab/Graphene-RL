#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.autograd import Variable
import logging
import matplotlib.pyplot as plt
from rdkit.Chem.rdmolfiles import MolFromPDBBlock
from rdkit.Chem import AllChem
from rdkit import DataStructs
from PIL import Image
import heapq
import cv2 as cv


def coord2contour_img(coords, img_fn=None):
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

    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    im = Image.fromarray(X)

    if img_fn is not None:
        im.save(img_fn)
        print("image saved to {}".format(img_fn))

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, frameon=False)
    # ax2.axis('off')
    # ax2.axis('equal')
    # ax2.imshow(X)
    # plt.show()

    return im

def cal_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def coord2feature(coord, n_features = 1024, cutoff = 5):
    # Step 1: Convert atom coordinates to pdb string block
    pdbStart = "CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1\n"
    pdbContent = "ATOM    ind  C   GRA X ind      xcoord  ycoord  96.500  0.00  0.00      SHT  C\n"
    pdbEnd = "END"
    ##############
    block = ''+pdbStart
    count = 0
    for atom in coord:
        pdbline = pdbContent
        count += 1
        x = atom[0]
        y = atom[1]
        block += pdbline.replace('ind',
                                str(count).rjust(3)).replace('xcoord', 
                                "{:.3f}".format(x).rjust(6)).replace('ycoord',
                                "{:.3f}".format(y).rjust(6))
    block += pdbEnd

    # Step 2: Get Mol object from PDB Block
    mol = MolFromPDBBlock(block)

    # Step 3: Convert Mol object to Morgan Fingerprints
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, cutoff, useFeatures=True, nBits = n_features)
    
    # Step 4: Convert Morgan Fingerprints to numpy array
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, array)
    array = array.reshape((1, -1))
    return array


def running_mean(x, window_size):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def plot_episode(rew_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(rew_list)), rew_list)
    plt.xlabel('timestep')
    plt.ylabel('accumulated reward')
    plt.savefig(save_path, bbox_inches='tight')
    # print('figure saved as {}'.format(save_path))
    plt.close()


def plot_training(acc_rewards, save_path):
    window_size = 10
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(acc_rewards)), acc_rewards)
    plt.plot(np.arange(window_size//2-1, len(acc_rewards)-window_size//2), running_mean(acc_rewards, 10), c='red')
    plt.xlabel('episode')
    plt.ylabel('accumulated reward')
    plt.savefig(save_path, bbox_inches='tight')
    # print('figure saved as {}'.format(save_path))
    plt.close()


def episode_finished(target_net, policy_net, env, episode, save_dir, start_time_str, acc_rewards):
    topk = max(1, int(len(acc_rewards)*0.1))
    top_rew = heapq.nlargest(topk, acc_rewards)
    rew_threshold = min(top_rew)

    if acc_rewards[-1] >= rew_threshold or (episode+1)%100 == 0:

        # plot timesteps vs. accumulated reward
        fn = '_'.join((start_time_str, 'ep', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        plot_episode(env.rew_list, save_path)

        # plot episode vs. accumulated reward
        fn = '_'.join((start_time_str, 'rew', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        plot_training(acc_rewards, save_path)

        # plot graphene structures in the episode
        fn = '_'.join((start_time_str, 'atoms', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        env.visualize_seq(save_path)

        # plot flux/rejection/remove atoms in the episode
        fn = '_'.join((start_time_str, 'flux', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        env.plot_flux(save_path)
        fn = '_'.join((start_time_str, 'ion_rej', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        env.plot_ion_rej(save_path)
        fn = '_'.join((start_time_str, 'num_rm', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        env.plot_remove_number(save_path)

        # save the target_net
        dqn_fn = '_'.join((start_time_str, 'target', str(episode))) + '.ckpt'
        dqn_path = os.path.join(save_dir, dqn_fn)
        torch.save(target_net.state_dict(), dqn_path)

        # save the policy_net
        dqn_fn = '_'.join((start_time_str, 'policy', str(episode))) + '.ckpt'
        dqn_path = os.path.join(save_dir, dqn_fn)
        torch.save(policy_net.state_dict(), dqn_path)

    return True


def episode_finished_dense(target_net, policy_net, env, episode, save_dir, start_time_str, acc_rewards):

    sub_save_dir = os.path.join(save_dir, str(episode))
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)

    # plot timesteps vs. accumulated reward
    fn = '_'.join((start_time_str, 'ep', str(episode))) + '.png'
    save_path = os.path.join(sub_save_dir, fn)
    plot_episode(env.rew_list, save_path)

    # plot graphene structures in the episode
    fn = '_'.join((start_time_str, 'atoms', str(episode))) + '.png'
    save_path = os.path.join(sub_save_dir, fn)
    env.visualize(save_path)

    # plot flux/rejection/remove atoms in the episode
    fn = '_'.join((start_time_str, 'flux', str(episode))) + '.png'
    save_path = os.path.join(sub_save_dir, fn)
    env.plot_flux(save_path)
    fn = '_'.join((start_time_str, 'ion_rej', str(episode))) + '.png'
    save_path = os.path.join(sub_save_dir, fn)
    env.plot_ion_rej(save_path)
    fn = '_'.join((start_time_str, 'num_rm', str(episode))) + '.png'
    save_path = os.path.join(sub_save_dir, fn)
    env.plot_remove_number(save_path)

    # save to pdb file
    fn = '_'.join((start_time_str, 'graphene', str(episode))) + '.pdb'
    save_path = os.path.join(sub_save_dir, fn)
    env.graphene.write2pdb(save_path)

    if (episode+1)%100 == 0:
        # plot episode vs. accumulated reward
        fn = '_'.join((start_time_str, 'rew', str(episode))) + '.png'
        save_path = os.path.join(save_dir, fn)
        plot_training(acc_rewards, save_path)

        # save the target_net
        dqn_fn = '_'.join((start_time_str, 'target', str(episode))) + '.ckpt'
        dqn_path = os.path.join(save_dir, dqn_fn)
        torch.save(target_net.state_dict(), dqn_path)

        # save the policy_net
        dqn_fn = '_'.join((start_time_str, 'policy', str(episode))) + '.ckpt'
        dqn_path = os.path.join(save_dir, dqn_fn)
        torch.save(policy_net.state_dict(), dqn_path)

    return True


def nearest_atom(atom_coord, removed_idx, num_atoms):
    if len(removed_idx) == 1:
        center = np.array(atom_coord[list(removed_idx)]).squeeze()
    else:
        center = np.array(atom_coord[list(removed_idx)]).mean(axis=0)
    dist_list = np.zeros(len(atom_coord))
    for i in range(len(atom_coord)):
        if i in removed_idx:
            dist_list[i] = 100
        else:
            dist_list[i] = cal_distance(center, atom_coord[i])
    nearest_idx = dist_list.argsort()[:num_atoms]
    return nearest_idx


def coord2area(coords):
    ########Coord to image##########
    # https://stackoverflow.com/questions/33094509/correct-sizing-of-markers-in-scatter-plot-to-a-radius-r-in-matplotlib
    figure = plt.figure(figsize=[5, 5])
    ax = plt.axes([0, 0, 1, 1], xlim=(0, 40), ylim=(0, 40))
    points_whole_ax = 5 *1 * 72    # 1 point = dpi / 72 pixels
    radius = 3.39/2
    points_radius = 2 * radius / (40) * points_whole_ax
    ax.scatter(coords[:,0], coords[:,1], s=points_radius**2, color='grey', edgecolors = 'k')
    ax.axis('off')
    plt.savefig('./process_img/temp.png')
    plt.close()
    ################Area from Image#############
    img = cv.imread('./process_img/temp.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
    ret,thresh = cv.threshold(gray,127,255,0)
    contours,hierarchy = cv.findContours(thresh, 
                                         cv.RETR_TREE, 
                                         cv.CHAIN_APPROX_NONE)
    print('Number of contours = ', str(len(contours)))
    ##############Get areas for all contours###########
    areas = [cv.contourArea(c) for c in contours]
    ###########Get the largest contour##########
    sort_index = np.argsort(areas)[::-1]
    ind = sort_index[0]
    ###########Scale conversion##################
    rate = (40**2)/(360**2)
    ###########################################
    final_area = areas[ind] * rate
    print('Area:', areas[ind] * rate)
    ##########Draw tht pore area contour(No need here)##########
#     cv.drawContours(img, contours[ind], -1, (0,255,0), 3)
#     cv.imshow('Image', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#     plt.show()
    os.remove('temp.png')
    return final_area