import numpy as np
import torch
import torch.nn as nn

from geomloss import SamplesLoss

import open3d as o3d


def chamfer_distance(x, y):
    x_ = torch.unsqueeze(x, dim=1)
    y_ = torch.unsqueeze(y, dim=2)
    distances = torch.sum((x_ - y_)**2, dim=3)
    # print(distances.shape)
    # print(torch.min(distances, dim=2)[1].shape)
    chamfer = torch.mean(torch.min(distances, dim=1)[0], dim=1) + torch.mean(torch.min(distances, dim=2)[0], dim=1)
    # print(chamfer.shape)
    return chamfer


def earth_mover_distance(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_EMD = 0
    # 近似计算
    L = SamplesLoss(loss='sinkhorn', p=1, blur=.05)
    for i in range(bs):
        loss = L(x[i], y[i])
        batch_EMD += loss
    emd = batch_EMD / bs
    return emd


class ChamferDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        cd = chamfer_distance(xyz1, xyz2)
        # print(cd)
        return cd
        # return torch.from_numpy(cd)


class EarthMoverDistance(nn.Module):
    def __init__(self):
        # super().__init__()
        super(EarthMoverDistance, self).__init__()

    def forward(self, xyz1, xyz2):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        emd = earth_mover_distance(xyz1, xyz2)
        return emd


CD = ChamferDistance()
EMD = EarthMoverDistance()


def cd_loss(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    return torch.mean(CD(pcs1, pcs2))


def emd_loss(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        pcs1 (torch.Tensor): (B, N, 3)
        pcs2 (torch.Tensor): (B, N, 3)
    """
    return torch.mean(EMD(pcs1, pcs2))



class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)

        return P

CD2 = ChamferLoss()


class Fs(nn.Module):
    def __init__(self):
        # super().__init__()
        super(Fs, self).__init__()

    def forward(self, xyz1, xyz2, th):
        """
        Args:
            xyz1: tensor with size of (B, N, 3)
            xyz2: tensor with size of (B, M, 3)
        """
        return self.t_f_s(xyz1, xyz2, th)

    def f_score(self, x, y, th=0.01):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        batch_f = 0

        for i in range(bs):
            pred, gt = x[i].detach().cpu().numpy(), y[i].detach().cpu().numpy()
            # pred, gt = x[i].cpu().numpy(), y[i].cpu().numpy()
            # pred, gt = x[i], y[i]

            pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
            gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))

            loss = 2 * recall * precision / (recall + precision) if recall + precision else 0

            batch_f += loss

        return torch.tensor(batch_f / bs)

    def t_f_s(self, x, y, th=0.1):
        bs, num_points, points_dim = x.size()

        x_ = torch.unsqueeze(x, dim=1)
        y_ = torch.unsqueeze(y, dim=2)
        distances = torch.sum((x_ - y_) ** 2, dim=3)
        # print(distances.shape)
        # print(torch.min(distances, dim=2)[1].shape)
        # chamfer = torch.mean(torch.min(distances, dim=1)[0], dim=1) + torch.mean(torch.min(distances, dim=2)[0], dim=1)

        d1 = torch.min(distances, dim=1)[0]
        d2 = torch.min(distances, dim=2)[0]

        d1[torch.where(d1.le(th))] = 0
        # d1[torch.where(d1.gt(th))] = 1

        d2[torch.where(d2.le(th))] = 0
        # d2[torch.where(d2.gt(th))] = 1

        chamfer = torch.mean(d1, dim=1) + torch.mean(d2, dim=1)

        return chamfer

        d1 = torch.sum(d1, dim=1)

        d2 = torch.sum(d2, dim=1)

        # return (d1.le(th).sum() + d2.le(th).sum()) / (2*bs*num_points)

        # d1.le(th).sum() + d2.le(th).sum()
        # print(d1.shape)

        # print(d1+d2)
        # print(2*num_points-(d1+d2))

        return torch.mean(d1+d2) / (2*num_points)



fs = Fs()


def fs_loss(xyz1, xyz2, th):

    return torch.mean(fs(xyz1, xyz2, th))



if __name__ == '__main__':

    pc1 = torch.rand(2, 2048, 3)
    pc2 = torch.rand(2, 2048, 3)

    print(fs(pc1, pc2, 0.001))


    import open3d as o3d

    source_points = pc1.numpy()[0]
    target_points = pc2.numpy()[0]

    print(source_points.shape)

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)


    dist1 = source.compute_point_cloud_distance(target)
    dist2 = target.compute_point_cloud_distance(source)
    print(np.array(dist1).shape)

    print(source_points)
    print(target_points)

    print(np.array(dist1))
    print(np.array(dist2))

    print((np.mean(np.array(dist1)**2) + np.mean(np.array(dist2)**2)))

    print(CD2(pc1, pc2))

    print(CD2(pc1, pc2).item()/2048)

    print(cd_loss(pc1, pc2).item())
    # print(emd_loss(pc1, pc2))]

