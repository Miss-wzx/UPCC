
import argparse
import copy
import os
# import time

import torch
from model.folding_net import AutoEncoder
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from metrics.loss import cd_loss, emd_loss
from history.history import History
from metrics.metric import chamfer_distance, earth_mover_distance, f_score
from model.pcn import PCN
import open3d as o3d

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
        print(len(self.mesh_filename))

    def __len__(self):
        return len(self.mesh_filename)

    def __getitem__(self, index):

        u_pcd = np.array(self.u_pcd[index], dtype=np.float32)
        c_pcd = np.array(self.c_pcd[index], dtype=np.float32)

        indices = np.random.choice(u_pcd.shape[0], self.n_points)
        u_pcd = u_pcd[indices]

        return u_pcd, c_pcd


def pcn_evaluate(args):
    print(args)
    # load model and evaluate
    with torch.no_grad():
        # device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = PcDataset(data_path, 512)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

        model = PCN(num_dense=2048, latent_dim=1024, grid_size=2, device=device).to(device)

        path = './model_save/pcn{}.pth'.format(opt)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print("model load weight done")
            model.eval()

        u_ = []
        c_ = []
        pred_ = []

        for batch_id, (u_pc, c_pc) in enumerate(dataloader, 0):
            u_pc = u_pc.to(device)
            # forward propagation
            coarse_pred, dense_pred = model(u_pc)

            u_.extend(np.array(u_pc.cpu().detach().numpy()))
            c_.extend(np.array(c_pc.cpu().detach().numpy()))
            pred_.extend(np.array(dense_pred.cpu().detach().numpy()))

        np.save('./app_res/' + args + '_u.npy', np.array(u_))
        np.save('./app_res/' + args + '_c.npy', np.array(c_))
        np.save('./app_res/' + args + '_pred.npy', np.array(pred_))
        print(np.array(pred_).shape)


def paun_evaluate(args):
    print(args)
    # load model and evaluate
    with torch.no_grad():
        # device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = PcDataset(data_path, 512)
        # dataset = PcDataset(data_path)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
        # model = UNet(input_num_pc=512, output_num_pc=2048, dropout=0., device=device).to(device)

        model = UNet(input_num_pc=512, output_num_pc=2048, dropout=0.).to(device)
        # model = UNet(num_pc=2048, dropout=0.).to(device)

        path = './model_save/paun{}.pth'.format(opt)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print("model load weight done")
            model.eval()

        u_ = []
        c_ = []
        pred_ = []

        for batch_id, (u_pc, c_pc) in enumerate(dataloader, 0):
            # 无序化
            u_pc, c_pc = u_pc[:, torch.randperm(u_pc.size()[1])], c_pc[:, torch.randperm(c_pc.size()[1])]

            u_pc, c_pc = u_pc.to(device), c_pc.to(device)

            # forward propagation
            # coarse_pred, dense_pred = model(u_pc, True)
            # coarse_pred, dense_pred = model(u_pc)

            dense_pred = model(u_pc)

            u_.extend(np.array(u_pc.cpu().detach().numpy()))
            c_.extend(np.array(c_pc.cpu().detach().numpy()))
            pred_.extend(np.array(dense_pred.cpu().detach().numpy()))

        np.save('./app_res/'+args+'_u.npy', np.array(u_))
        np.save('./app_res/'+args+'_c.npy', np.array(c_))
        np.save('./app_res/'+args+'_pred.npy', np.array(pred_))


def evaluate(args):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device('cuda')
        dataset = PcDataset('./data/data_m3_1155.pkl', 512)
        # dataset = PcDataset(data_path)

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

        # model = UNet(input_num_pc=512, output_num_pc=2048, dropout=0., device=device).to(device)
        model = UNet(input_num_pc=512, output_num_pc=2048, dropout=0.).to(device)

        # model = UNet(num_pc=2048, dropout=0.).to(device)

        path = './model_save/paun{}.pth'.format(opt)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print("model load weight done")
            model.eval()

        dataloader_iter = iter(dataloader)
        u, c = next(dataloader_iter)
        u, c = u.to(device), c.to(device)
        np.save("patuu_t.npy", np.array(u.cpu().detach().numpy()))

        np.save("patuc_t.npy", np.array(c.cpu().detach().numpy()))

        # coarse_pred, dense_pred = model(u)
        dense_pred = model(u)

        print(dense_pred.shape)
        np.save("patupred_t.npy", np.array(dense_pred.cpu().detach().numpy()))


def fn_evaluate(args):
    print(args)
    # load model and evaluate
    with torch.no_grad():
        # device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        dataset = PcDataset(data_path, 512)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

        model = AutoEncoder()
        model.to(device)

        path = './model_save/folding_net{}.pth'.format(opt)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print("model load weight done")
            model.eval()

        u_ = []
        c_ = []
        pred_ = []

        for batch_id, (u_pc, c_pc) in enumerate(dataloader, 0):
            p = u_pc.to(device)
            p = p.permute(0, 2, 1)
            # p = p.to(device)

            dense_pred = model(p)
            dense_pred = dense_pred.permute(0, 2, 1)

            u_.extend(np.array(u_pc.cpu().detach().numpy()))
            c_.extend(np.array(c_pc.cpu().detach().numpy()))
            pred_.extend(np.array(dense_pred.cpu().detach().numpy()))

        np.save('./app_res/' + args + '_u.npy', np.array(u_))
        np.save('./app_res/' + args + '_c.npy', np.array(c_))
        np.save('./app_res/' + args + '_pred.npy', np.array(pred_))


def cd_emd_fs_eval(el):
    for name in el:
        print(name)
        s = './app_res/{}'.format(name)

        p = np.load("{}_u.npy".format(s))
        c = np.load("{}_c.npy".format(s))
        dense_pred = np.load("{}_pred.npy".format(s))

        c = torch.from_numpy(c)
        dense_pred = torch.from_numpy(dense_pred)

        cd = chamfer_distance(dense_pred, c)
        emd = earth_mover_distance(dense_pred, c)
        fs1 = f_score(dense_pred, c, th=0.01)

        fs2 = f_score(dense_pred, c, th=0.02)

        fs3 = f_score(dense_pred, c, th=0.05)

        print('cd: ', cd, 'emd: ', emd, 'fs: ', fs1, fs2, fs3)


def test_app():
    d = []
    for name in os.listdir('./data/test_data_pcd/'):
        pcd = o3d.io.read_point_cloud('./data/test_data_pcd/'+name)
        pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()), center=pcd.get_center())
        pcd = copy.deepcopy(pcd).translate((0., 0., 0.), relative=False)

        pcd = np.array(pcd.points)
        indices = np.random.choice(pcd.shape[0], 2048)
        pcd = pcd[indices]
        d.append(pcd)

    np.save("./app_res/test_data_pcd_c.npy", np.array(d, dtype=np.float32))

    with torch.no_grad():
        device = torch.device('cuda')

        model = UNet(num_pc=2048, dropout=0.).to(device)

        if os.path.exists('./model_save/paun0.pth'):
            model.load_state_dict(torch.load('./model_save/paun0.pth', map_location=device))
            print("model load weight done")
            model.eval()

        u = torch.from_numpy(np.array(d, dtype=np.float32)).to(device)

        dense_pred = model(u)
        print(dense_pred.shape)
        np.save("./app_res/test_data_pcd_pred.npy", np.array(dense_pred.cpu().detach().numpy()))


data_path = './data/data_test.pkl'
# data_path = './data/data.pkl'
# data_path = './data/data_m3_1155_test.pkl'

# opt = '_opt'
opt = '_lw_1155_15'
# opt = ''

# opt = '_2k'
# opt = '_1155_1'

if __name__ == '__main__':
    # evaluate('')
    paun_evaluate('paun')
    # pcn_evaluate('pcn')
    # fn_evaluate('fn')

    # test_app()
    # cd_emd_fs_eval(['paun'])
    # cd_emd_fs_eval(['pcn', 'fn'])

    # cd_emd_fs_eval(['pcn', 'fn', 'paun'])
