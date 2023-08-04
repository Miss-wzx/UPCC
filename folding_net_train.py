import argparse
# import os
# import time

import torch
import torch.optim as optim
from model.folding_net import AutoEncoder
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from metrics.loss import cd_loss
from history.history import History
from metrics.metric import chamfer_distance, earth_mover_distance, f_score


class PcDataset(Dataset):
    def __init__(self, data_path, n_points=2048):
        self.data_path = data_path
        self.n_points = n_points

        h = pickle.loads(open(self.data_path, 'rb').read())

        self.mesh_filename = h['mesh_filename']
        self.u_pcd = h['u_pcd']
        self.c_pcd = h['c_pcd']

        print(self.mesh_filename)
        r = []
        for i in range(len(self.mesh_filename)):
            if len(self.u_pcd[i]) < 5:
                r.append(i)
        print(r)
        l = 0
        for i in r:
            self.mesh_filename.pop(i - l)
            self.u_pcd.pop(i - l)
            self.c_pcd.pop(i - l)
            l += 1
        print(len(self.mesh_filename))

    def __len__(self):
        return len(self.mesh_filename)

    def __getitem__(self, index):

        u_pcd = np.array(self.u_pcd[index], dtype=np.float32)
        c_pcd = np.array(self.c_pcd[index], dtype=np.float32)

        indices = np.random.choice(u_pcd.shape[0], self.n_points)
        u_pcd = u_pcd[indices]

        return u_pcd, c_pcd


def test_model(model, data_loader, device):
    model = model.eval()
    cd_sum = []
    emd_sum = [0]
    fs_sum = []
    loss_sum = []

    for batch_id, (p, c) in enumerate(data_loader, 0):

        p = p.permute(0, 2, 1)
        p = p.to(device)
        # c = c.permute(0, 2, 1)
        c = c.to(device)
        recons = model(p)
        recons = recons.permute(0, 2, 1)
        loss = cd_loss(c, recons)

        loss_sum.append(loss.item())

        cd_sum.append(chamfer_distance(recons, c))
        # emd_sum.append(earth_mover_distance(recons, c))
        fs_sum.append(f_score(recons, c, th=0.01))

    return sum(loss_sum)/len(loss_sum), sum(cd_sum)/len(cd_sum), sum(emd_sum)/len(emd_sum), sum(fs_sum)/len(fs_sum)


def train(args):
    # 读取数据
    train_dataset = PcDataset('./data/data_m3_1155.pkl', args.n_points)
    # test_dataset = PcDataset('./data/data_test.pkl', args.n_points)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model
    autoendocer = AutoEncoder()
    autoendocer.to(device)

    # optimizer
    optimizer = optim.Adam(autoendocer.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    scheduler_interval = 200

    h = History('folding_net', './history', args.epochs, args.batch_size, info='_1155')

    # 加载模型参数断点续训

    # 训练模型
    best_test_cd_loss = 100

    for epoch in range(args.epochs):

        print('[%d/%d: ' % (epoch + 1, args.epochs) + '-' * 30 + ' ]')

        autoendocer.train()
        train_loss_sum = 0

        for batch_id, (p, c) in enumerate(train_dataloader, 0):
            p, c = p.to(device), c.to(device)
            p = p.permute(0, 2, 1)
            p = p.to(device)
            c = c.to(device)
            recons = autoendocer(p)
            ls = cd_loss(c, recons.permute(0, 2, 1))

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            train_loss_sum += ls.item()

            print('[%d/%d: %d/%d] loss: %f' % (
                epoch + 1, args.epochs, batch_id + 1, len(train_dataloader), ls.item()))

        if (epoch+1) % scheduler_interval == 0:
            scheduler.step()

        print('[%d/%d: loss: %f]' % (epoch + 1, args.epochs, train_loss_sum / len(train_dataloader)))

        h.train_loss.append(train_loss_sum / len(train_dataloader))
        h.train_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # h.save_history()

        # if (epoch + 1) % 10 == 0:
        #
        #     loss_test, cd_test, emd_test, fs_test = test_model(autoendocer, test_dataloader, device)
        #
        #     print('test model: ', loss_test, cd_test, emd_test, fs_test)
        #
        #     h.test_loss.append(loss_test)
        #     h.test_cd.append(cd_test)
        #     h.test_emd.append(emd_test)
        #     h.test_fs.append(fs_test)
        #
        #     if loss_test < best_test_cd_loss:
        #         print('保存模型')
        #         best_test_cd_loss = loss_test
        #         torch.save(autoendocer.cpu().state_dict(), './model_save/folding_net_opt.pth')
        #         autoendocer.to(device)

        h.save_history()

    # 保存网络模型
    torch.save(autoendocer.cpu().state_dict(), './model_save/folding_net_1155.pth')
    h.save_history()


def main():
    # PARAMETERS
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_points', type=int, default=512)
    parser.add_argument('--m_points', type=int, default=2025)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
