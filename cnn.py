import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader

from dataset import GraphenImageDataset


class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, in_dim, bias=False),
        )

    def forward(self, x):
        return x + self.main(x)


class MLP(nn.Module):
    def __init__(self, in_size):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            ResidualBlock(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            ResidualBlock(64),
            nn.ReLU(),
        )
        self.output = nn.Linear(64, 1)
    
    def forward(self, x):
        out = self.model(x)
        return self.output(out), out


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    # torch.cuda.empty_cache()

    batch_size = 64
    lr_mlp = 0.001
    lr_resnet = 0.0001
    num_epochs = 600
    obj = 'flux'

    transform = transforms.Compose([
        transforms.CenterCrop((380, 380)),
        transforms.Resize((224, 224))
    ])
    train_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path='./data/id_prop.csv', 
        transform=transform, 
        mode='train',
        label_mode=obj
    )
    test_dataset = GraphenImageDataset(
        img_dir='./data/image', 
        csv_path='./data/id_prop.csv', 
        transform=transform, 
        mode='test',
        label_mode=obj
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print('training set size:', len(train_dataset))
    print('testing set size:', len(test_dataset))

    resnet = torchvision.models.resnet18(pretrained=True).to(device)
    mlp = MLP(in_size=1000).to(device)
    print(resnet)
    print(mlp)

    ct = 0
    for child in resnet.children():
        ct += 1
        if ct < 6:
            for param in child.parameters():
                param.requires_grad = False

    loss_func = nn.MSELoss(reduction='mean')
    # loss_func = nn.SmoothL1Loss(reduction='mean')
    optim_mlp = Adam(mlp.parameters(), lr=lr_mlp)
    optim_resnet = Adam(resnet.parameters(), lr=lr_resnet)

    mlp_lr_scheduler = lr_scheduler.MultiStepLR(optim_mlp, gamma=0.1, milestones=[400])
    resnet_lr_scheduler = lr_scheduler.MultiStepLR(optim_resnet, gamma=0.1, milestones=[400])

    train_losses, test_losses = [], []
    for epoch in range(num_epochs):

        acc_train_loss = 0.0
        resnet.train()
        mlp.train()
        for i, (img, label) in enumerate(train_dataloader):
            img, label = img.to(device), label.to(device)
            if len(label.shape) == 1:
                label = torch.unsqueeze(label, 1)
            
            feat = resnet(img)
            pred, __ = mlp(feat)

            optim_mlp.zero_grad()
            optim_resnet.zero_grad()
            loss = loss_func(pred, label)
            loss.backward()
            acc_train_loss += loss.item()
            optim_mlp.step()
            optim_resnet.step()

            torch.cuda.empty_cache()

        train_losses.append(acc_train_loss/(i+1))

        # lr decay
        mlp_lr_scheduler.step()
        resnet_lr_scheduler.step()

        # validation on test data
        resnet.eval()
        mlp.eval()
        predictions = np.zeros(len(test_dataset))
        labels = np.zeros(len(test_dataset))
        start_idx, end_idx = 0, 0
        acc_test_loss = 0.0
        # with torch.no_grad():
        for i, (img, label) in enumerate(test_dataloader):
            img, label = img.to(device), label.to(device)
            if len(label.shape) == 1:
                label = torch.unsqueeze(label, 1)
            batch_size = label.shape[0]
            end_idx += batch_size
            
            feat = resnet(img)
            pred, __ = mlp(feat)

            loss = loss_func(pred, label)
            acc_test_loss += loss.item()

            if device is 'cpu':
                pred = pred.detach().numpy()
                label = label.detach().numpy()
            else:
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()

            predictions[start_idx:end_idx] = np.squeeze(pred)
            labels[start_idx:end_idx] = np.squeeze(label)
            start_idx = end_idx
                
        test_losses.append(acc_test_loss/(i+1))
        
        print("epoch: {}, training Loss: {}, testing loos: {}".format(
            epoch, train_losses[-1], test_losses[-1]))

        # print("epoch: {}, training Loss: {}".format(epoch, train_losses[-1]))

    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(mlp.state_dict(), os.path.join(model_dir, 'mlp_new_{}.ckpt'.format(obj)))
    torch.save(resnet.state_dict(), os.path.join(model_dir, 'resnet_new_{}.ckpt'.format(obj)))
    # torch.save(cnn.state_dict(), os.path.join(model_dir, 'cnn.ckpt'))

    resnet.eval()
    mlp.eval()

    # validation on train data
    train_predictions = np.zeros(len(train_dataset))
    train_labels = np.zeros(len(train_dataset))
    start_idx, end_idx = 0, 0
    for i, (img, label) in enumerate(train_dataloader):
        img, label = img.to(device), label.to(device)
        if len(label.shape) == 1:
            label = torch.unsqueeze(label, 1)
        batch_size = label.shape[0]
        end_idx += batch_size
        
        feat = resnet(img)
        pred, __ = mlp(feat)

        if device is 'cpu':
            pred = pred.detach().numpy()
            label = label.detach().numpy()
        else:
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

        train_predictions[start_idx:end_idx] = np.squeeze(pred)
        train_labels[start_idx:end_idx] = np.squeeze(label)
        start_idx = end_idx

    # validation on test data
    # test_predictions = np.zeros(len(test_dataset))
    # test_labels = np.zeros(len(test_dataset))
    # start_idx, end_idx = 0, 0
    # for i, (img, label) in enumerate(train_dataloader):
    #     img, label = img.to(device), label.to(device)
    #     if len(label.shape) == 1:
    #         label = torch.unsqueeze(label, 1)
    #     batch_size = label.shape[0]
    #     end_idx += batch_size
    #     print(batch_size, start_idx, end_idx)
        
    #     feat = resnet(img)
    #     pred = mlp(feat)

    #     if device is 'cpu':
    #         pred = pred.detach().numpy()
    #         label = label.detach().numpy()
    #     else:
    #         pred = pred.detach().cpu().numpy()
    #         label = label.detach().cpu().numpy()

    #     test_predictions[start_idx:end_idx] = np.squeeze(pred)
    #     test_labels[start_idx:end_idx] = np.squeeze(label)
    #     start_idx = end_idx

    test_predictions = np.zeros(len(test_dataset))
    test_labels = np.zeros(len(test_dataset))

    start_idx, end_idx = 0, 0
    for i, (img, label) in enumerate(test_dataloader):
        img, label = img.to(device), label.to(device)
        if len(label.shape) == 1:
            label = torch.unsqueeze(label, 1)
        batch_size = label.shape[0]
        end_idx += batch_size

        feat = resnet(img)
        pred, __ = mlp(feat)

        if device is 'cpu':
            pred = pred.detach().numpy()
            label = label.detach().numpy()
        else:
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()

        test_predictions[start_idx:end_idx] = np.squeeze(pred)
        test_labels[start_idx:end_idx] = np.squeeze(label)
        start_idx = end_idx

    print("MSE on training set:", np.mean(np.square(train_predictions - train_labels)))
    print("MSE on testing set:", np.mean(np.square(test_predictions - test_labels)))
    print("L1 error on training set:", np.mean(np.abs(train_predictions - train_labels)))
    print("L1 error on testing set:", np.mean(np.abs(test_predictions - test_labels)))


    plot_dir = './plot'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    x = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
    y = np.linspace(np.min(test_dataset.labels), np.max(train_dataset.labels))
    plt.figure()
    plt.scatter(train_predictions, train_labels, c='blue', marker='x')
    plt.scatter(predictions, labels, c='red', marker='x')
    plt.plot(x, y, linestyle='dashed', c='black')
    plt.xlabel('prediction')
    plt.ylabel('label')
    plt.title('Flux prediction')
    # plt.show()
    plt.savefig(os.path.join(plot_dir, 'pred_new_{}.png'.format(obj)))

    # print(train_losses)
    # print(test_losses)
    plt.figure()
    plt.plot(train_losses, c='blue')
    plt.plot(test_losses, c='red')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'testing loss'])
    plt.savefig(os.path.join(plot_dir, 'loss_new_{}.png'.format(obj)))


if __name__ == "__main__":
    train()

    # device = 'cpu'
    # resnet = torchvision.models.resnet(pretrained=False).to(device)
    # mlp = MLP(in_size=1000).to(device)

    # model_dir='./models'
    # resnet.load_state_dict(torch.load(
    #     os.path.join(model_dir, 'resnet_flux.ckpt'), map_location=device
    # ))
    # mlp.load_state_dict(torch.load(
    #     os.path.join(model_dir, 'mlp_flux.ckpt'), map_location=device
    # ))

    # from PIL import Image

    # transform = transforms.Compose([
    #     transforms.CenterCrop((500, 500)),
    #     transforms.Resize((224, 224))
    # ])

    # img = Image.open('./data/img/1_60.png')
    # img = img.convert('RGB')
    # img = transform(img)
    # img = np.rollaxis(np.array(img), 2, 0)

    # img = torch.from_numpy(img).type(torch.FloatTensor)
    # img = img.unsqueeze(0)

    # print(mlp(resnet(img)))