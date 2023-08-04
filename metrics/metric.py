import torch
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def chamfer_distance(x, y):
    x_ = torch.unsqueeze(x, dim=1)
    y_ = torch.unsqueeze(y, dim=2)
    distances = torch.sum((x_ - y_)**2, dim=3)
    chamfer = torch.mean(torch.min(distances, dim=1)[0], dim=1) + torch.mean(torch.min(distances, dim=2)[0], dim=1)
    return float(torch.mean(chamfer))


def emd(X, Y):
    d = cdist(X, Y)
    # print(d)
    assignment = linear_sum_assignment(d)
    # print(assignment)
    return d[assignment].sum() / min(len(X), len(Y))


def earth_mover_distance(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_emd = 0
    for i in range(bs):
        pred, gt = x[i].detach().cpu().numpy(), y[i].detach().cpu().numpy()
        batch_emd += emd(pred, gt)
    return batch_emd / bs


def f_score(x, y, th=0.01):

    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_f = 0

    for i in range(bs):
        pred, gt = x[i].detach().cpu().numpy(), y[i].detach().cpu().numpy()
        pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
        gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

        dist1 = pred.compute_point_cloud_distance(gt)
        dist2 = gt.compute_point_cloud_distance(pred)

        recall = float(sum(d < th for d in dist2)) / float(len(dist2))
        precision = float(sum(d < th for d in dist1)) / float(len(dist1))

        loss = 2 * recall * precision / (recall + precision) if recall + precision else 0

        batch_f += loss

    return batch_f / bs


if __name__ == '__main__':
    pc1 = torch.rand(2, 2048, 3)
    pc2 = torch.rand(2, 2048, 3)
    print(pc1.size())
    print(chamfer_distance(pc1, pc2))
    print(earth_mover_distance(pc1, pc2))
    print(f_score(pc1, pc2, th=0.01))

