import os
import pickle
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class GraphenImageDataset(Dataset):
    def __init__(self, 
        img_dir='./data/img',
        model_dir='./models', 
        csv_path=None, 
        transform=None, 
        mode='train', 
        label_mode='both',
        standardize=True
    ):
        super(GraphenImageDataset, self).__init__()
        self.transform = transform
        # self.mode = mode
        imgs, labels = [], []
        self.fn_list = []
        self.test_img, self.test_label = [], []

        with open(csv_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(',')
                img_fn = line[0] + '.png'
                img = Image.open(os.path.join(img_dir, img_fn))
                img = img.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                img = np.rollaxis(np.array(img), 2, 0) / 255.0
                imgs.append(img)
                labels.append([float(line[1]), float(line[2])])
                self.fn_list.append(line[0])

                if '1_' in line[0] and len(self.test_img) < 10:
                    self.test_img.append(img)
                    self.test_label.append(float(line[2]))

        np.random.seed(0)
        num_data = len(labels)
        test_idx = np.random.choice(num_data, num_data//5, replace=False).tolist()
        train_idx = list(set(range(num_data)) - set(test_idx))

        imgs = np.array(imgs)
        labels = np.array(labels)

        if standardize:
            # data standardize
            flux_mean, flux_std = np.mean(labels[:,0]), np.std(labels[:,0])
            rej_mean, rej_std = np.mean(labels[:,1]), np.std(labels[:,1])
            labels[:,0] = (labels[:,0] - flux_mean) / flux_std
            labels[:,1] = (labels[:,1] - rej_mean) / rej_std

        if mode is 'train':
            imgs = np.array(imgs)
            labels = np.array(labels)
            imgs = imgs[train_idx, :, :]
            labels = labels[train_idx, :]
        elif mode is 'test':
            imgs = np.array(imgs)
            labels = np.array(labels)
            imgs = imgs[test_idx, :, :]
            labels = labels[test_idx, :]
        elif mode is 'all':
            imgs = np.array(imgs)
            labels = np.array(labels)
        else:
            raise ValueError('mode must be train, test or all')

        # # data standardize
        # flux_mean, flux_std = np.mean(labels[:,0]), np.std(labels[:,0])
        # rej_mean, rej_std = np.mean(labels[:,1]), np.std(labels[:,1])
        # labels[:,0] = (labels[:,0] - flux_mean) / flux_std
        # labels[:,1] = (labels[:,1] - rej_mean) / rej_std

        self.label_mode = label_mode
        self.imgs = imgs
        self.labels = labels
        # self.imgs = torch.from_numpy(self.imgs).type(torch.FloatTensor)
        # self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)

        if standardize:
            self.mean_std = {
                'flux_mean': flux_mean, 
                'flux_std': flux_std, 
                'rej_mean': rej_mean, 
                'rej_std': rej_std
            }
            with open(os.path.join(model_dir, 'mean_std.pickle'), 'wb') as handle:
                pickle.dump(self.mean_std, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(self.mean_std)


    def __getitem__(self, index):
        img = torch.from_numpy(self.imgs[index]).type(torch.FloatTensor)
        if self.label_mode is 'both':
            label = torch.from_numpy(self.labels[index]).type(torch.FloatTensor)
        elif self.label_mode is 'flux':
            label = torch.from_numpy(self.labels[index]).type(torch.FloatTensor)[0]
        elif self.label_mode is 'rej':
            label = torch.from_numpy(self.labels[index]).type(torch.FloatTensor)[1]
        return img, label


    def __len__(self):
        return len(self.labels)


    def plot_prop(self, save_path=None):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.scatter(self.labels[:,0], self.labels[:,1], c='red')
        plt.xlabel('flux')
        plt.ylabel('ion rejection')
        plt.title('Dataset distribution')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)


    def get_test_case(self):        
        img = torch.from_numpy(self.test_img).type(torch.FloatTensor)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        return img, self.test_label



if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.CenterCrop((360, 360)),
        transforms.Resize((224, 224))
    ])
    dataset = GraphenImageDataset(transform=transform, mode='all')

    dataset.get_test_case()