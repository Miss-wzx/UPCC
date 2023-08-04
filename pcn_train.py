# 导入相应库
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
import argparse
from model.pcn import PCN
from metrics.loss import cd_loss, emd_loss
from history.history import History
from metrics.metric import chamfer_distance, earth_mover_distance, f_score


def parse_args():
    # PARAMETERS
    parser = argparse.ArgumentParser('pcn')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    return parser.parse_args()


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


def test_model(model, data_loader, alpha, device):
    model = model.eval()
    cd_sum = []
    emd_sum = []
    fs_sum = []
    loss_sum = []

    for batch_id, (p, c) in enumerate(data_loader, 0):
        p, c = p.to(device), c.to(device)

        coarse_pred, dense_pred = model(p)

        # loss function
        loss1 = cd_loss(coarse_pred, c)
        loss2 = cd_loss(dense_pred, c)
        loss = loss1 + alpha * loss2

        loss_sum.append(loss.item())

        cd_sum.append(chamfer_distance(dense_pred, c))
        emd_sum.append(earth_mover_distance(dense_pred, c))
        fs_sum.append(f_score(dense_pred, c, th=0.01))

    return sum(loss_sum)/len(loss_sum), sum(cd_sum)/len(cd_sum), sum(emd_sum)/len(emd_sum), sum(fs_sum)/len(fs_sum)


def main(p_args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 读取数据
    train_dataset = PcDataset('./data/data_m3_1155.pkl', 512)
    # test_dataset = PcDataset('./data/data_test.pkl', 512)

    train_data_loader = DataLoader(train_dataset, batch_size=p_args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=p_args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    h = History('pcn', './history', p_args.epoch, p_args.batch_size, info='_1155_1')

    # 实例化模型
    model = PCN(num_dense=2048, latent_dim=1024, grid_size=2, device=device).to(device)

    # 加载模型参数断点续训

    # 优化器选择和动态学习率
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.7)

    # 训练模型
    best_test_cd_loss = 100

    for epoch in range(p_args.epoch):

        # hyperparameter alpha 最好训练批次的10倍
        if epoch < 50:
            alpha = 0.01
        elif epoch < 100:
            alpha = 0.1
        elif epoch < 300:
            alpha = 0.5
        else:
            alpha = 1.0

        print('[%d/%d: ' % (epoch+1, p_args.epoch) + '-'*30 + ' ]')
        print(alpha)

        model.train()
        train_loss_sum = 0

        for batch_id, (p, c) in enumerate(train_data_loader, 0):
            p, c = p.to(device), c.to(device)

            optimizer.zero_grad()

            # forward propagation
            coarse_pred, dense_pred = model(p)

            # loss function
            loss1 = cd_loss(coarse_pred, c)
            loss2 = cd_loss(dense_pred, c)
            loss3 = emd_loss(dense_pred, c)

            loss = loss1 + alpha * loss2 + alpha *loss3

            train_loss_sum += loss.item()

            # back propagation
            loss.backward()
            optimizer.step()

            print('[%d/%d: %d/%d] loss: %f' % (
                epoch + 1, p_args.epoch, batch_id + 1, len(train_data_loader), loss.item()))

        scheduler.step()
        print('[%d/%d: loss: %f]' % (epoch + 1, p_args.epoch, train_loss_sum / len(train_data_loader)))

        h.train_loss.append(train_loss_sum / len(train_data_loader))
        h.train_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # h.save_history()

        # if (epoch+1) % 10 == 0:
        #
        #     loss_test, cd_test, emd_test, fs_test = test_model(model, test_data_loader, alpha, device)
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
        #         torch.save(model.cpu().state_dict(), './model_save/pcn_opt.pth')
        #         model.to(device)

        h.save_history()

    # 保存网络模型
    torch.save(model.cpu().state_dict(), './model_save/pcn_1155_1.pth')
    h.save_history()


if __name__ == '__main__':
    args = parse_args()
    main(args)
