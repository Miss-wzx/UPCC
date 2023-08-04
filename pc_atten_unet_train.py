import argparse
import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from model.pc_atten_unet import UNet
from model.Scheduler import GradualWarmupScheduler
from metrics.loss import cd_loss, emd_loss, fs_loss
from history.history import History
from metrics.metric import chamfer_distance, earth_mover_distance, f_score

from model.pc_atten_unet import UNet


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
            self.mesh_filename.pop(i-l)
            self.u_pcd.pop(i-l)
            self.c_pcd.pop(i-l)
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
    with torch.no_grad():
        model = model.eval()
        cd_sum = []
        emd_sum = []
        fs_sum = []
        loss_sum = []

        for batch_id, (p, c) in enumerate(data_loader, 0):
            p, c = p.to(device), c.to(device)
            # forward propagation
            dense_pred = model(p)

            # loss function
            loss1 = cd_loss(dense_pred, c)
            loss2 = emd_loss(dense_pred, c)
            loss = loss1 + alpha * loss2

            loss_sum.append(loss.item())
            emd_sum.append(loss2.item())

            cd_sum.append(chamfer_distance(dense_pred, c))
            # emd_sum.append(earth_mover_distance(dense_pred, c))
            fs_sum.append(f_score(dense_pred, c, th=0.01))

    return sum(loss_sum)/len(loss_sum), sum(cd_sum)/len(cd_sum), sum(emd_sum)/len(emd_sum), sum(fs_sum)/len(fs_sum)


def train(args):
    device = torch.device(args.device)
    # dataset
    # dataset = PcDataset('./data/data.pkl', 512)
    dataset = PcDataset('./data/data_m3_1155.pkl', 512)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    # test_dataset = PcDataset('./data/data_test.pkl')
    # test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    # model setup
    net_model = UNet(input_num_pc=512, output_num_pc=2048, dropout=args.dropout).to(device)

    # optimizer = torch.optim.AdamW(net_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    # cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0, last_epoch=-1)
    # warm_up_scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=args.multiplier, warm_epoch=args.epoch//20, after_scheduler=cosine_scheduler)

    optimizer = optim.Adam(net_model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)
    warm_up_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

    h = History('paun', './history', args.epoch, args.batch_size, info='_lw_1155_15')
    # h = pickle.loads(open('./history/paun_e1000.pkl', 'rb').read())

    start_epoch = 0
    temp_loss = 100

    # if os.path.exists('./model_save/paun_7k.pth'):
    #     net_model.load_state_dict(torch.load('./model_save/paun_7k.pth', map_location=device))
    #     # optimizer.load_state_dict(torch.load('./model_save/paun_7k.pth'))  # 加载优化器参数
    #
    #     print("model load weight done")
    #
    # for i in range(start_epoch):
    #     for j in range(len(dataloader)):
    #         optimizer.zero_grad()
    #         optimizer.step()
    #     warm_up_scheduler.step()

    # start training
    for epoch in range(start_epoch, args.epoch):
        print('[%d/%d: ' % (epoch+1, args.epoch) + '-'*30 + ' ]')
        loss_sum = 0

        # if epoch < 300:
        #     alpha = 0.1
        # else:
        #     alpha = 0.1

        # hyperparameter alpha 最好训练批次的10倍
        if epoch < 100:
            alpha = 0.5
        elif epoch < 300:
            alpha = 1.0
        else:
            alpha = 1.5

        # alpha = 0.5
        print(alpha)

        for batch_id, (u_pc, c_pc) in enumerate(dataloader, 0):
            optimizer.zero_grad()
            # 无序化
            u_pc, c_pc = u_pc[:, torch.randperm(u_pc.size()[1])], c_pc[:, torch.randperm(c_pc.size()[1])]
            # u_pc = u_pc[:, torch.randperm(u_pc.size()[1])]

            u_pc, c_pc = u_pc.to(device), c_pc.to(device)

            # forward propagation
            dense_pred = net_model(u_pc)

            # loss function
            loss0 = cd_loss(dense_pred, c_pc)

            loss3 = emd_loss(dense_pred, c_pc)

            loss = loss0 + alpha * loss3

            # if epoch < 200:
            #
            #     print(loss1.item())
            #     loss2 = fs_loss(dense_pred, c_pc, loss1.item() / ((epoch + 100) / 100))
            #     # loss2 = fs_loss(dense_pred, c_pc, loss1.item())
            #
            #     print(loss2.item())
            #
            #     loss = loss0 + loss1 + alpha * loss2
            #
            # else:
            #     b = 0.5
            #     print(b)
            #     print(loss1.item())
            #     loss2 = fs_loss(dense_pred, c_pc, loss1.item() / ((epoch + 100) / 100))
            #     # loss2 = fs_loss(dense_pred, c_pc, loss1.item())
            #
            #     print(loss2.item())
            #
            #     loss3 = emd_loss(dense_pred, c_pc)
            #
            #     loss = loss0 + loss1 + alpha * loss2 + b*loss3

            # loss = emd_loss(dense_pred, c_pc)
            # print(loss1.item())
            # loss2 = fs_loss(dense_pred, c_pc, loss1.item()/((epoch+100)/100))
            # # loss2 = fs_loss(dense_pred, c_pc, loss1.item())
            #
            # print(loss2.item())
            #
            # loss = loss1 + alpha * loss2

            # loss = loss2

            # print(loss)
            # back propagation
            loss.backward()

            loss_sum += loss.item()
            optimizer.step()

            print('[%d/%d: %d/%d] loss: %f LR: %f' % (
                epoch + 1, args.epoch, batch_id + 1, len(dataloader), loss.item(), optimizer.state_dict()['param_groups'][0]['lr']))

        warm_up_scheduler.step()

        h.train_loss.append(loss_sum / len(dataloader))
        h.train_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
        # h.save_history()

        # if (epoch + 1) % 10 == 0:
        #
        #     loss_test, cd_test, emd_test, fs_test = test_model(net_model, test_data_loader, alpha, device)
        #
        #     print('test model: ', loss_test, cd_test, emd_test, fs_test)
        #
        #     h.test_loss.append(loss_test)
        #     h.test_cd.append(cd_test)
        #     h.test_emd.append(emd_test)
        #     h.test_fs.append(fs_test)
        #
        # if loss_sum / len(dataloader) < temp_loss:
        #     print('保存模型')
        #     temp_loss = loss_sum / len(dataloader)
        #     torch.save(net_model.cpu().state_dict(), './model_save/paun_opt.pth')
        #     net_model.to(device)

        torch.save(net_model.cpu().state_dict(), './model_save/paun_lw_1155_15.pth')
        net_model.to(device)

        h.save_history()

    torch.save(net_model.cpu().state_dict(), './model_save/paun_lw_1155_15.pth')


def main():
    # PARAMETERS
    parser = argparse.ArgumentParser('paun')
    parser.add_argument('--device', default='cuda', type=str, help='use gpu/cpu mode')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size in training')
    parser.add_argument('--epoch', default=1000, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--multiplier', default=1.1, type=float, help='multiplier')
    parser.add_argument('--dropout', default=0.10, type=float, help='dropout')
    parser.add_argument('--optimizer', default='AdamW', type=str, help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
