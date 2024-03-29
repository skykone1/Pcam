"""
The code for the optional safeguard and refinement by optimisation or icp, proposed by DGR [1-3], are taken from
https://github.com/chrischoy/DeepGlobalRegistration/blob/master

[1] Christopher Choy, Wei Dong, Vladlen Koltun. Deep Global Registration, CVPR, 2020.
[2] Christopher Choy, Jaesik Park, Vladlen Koltun. Fully Convolutional Geometric Features. ICCV, 2019.
[3] Christopher Choy, JunYoung Gwak, Silvio Savarese. 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. CVPR, 2019.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='5'
import time
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch.optim as optim
import MinkowskiEngine as ME
import torch.nn.functional as F
import lightconvpoint.nn as lcp_nn
from pcam.tool.log import Logger, save_metrics
from pcam.tool.loss import compute_metrics
from pcam.tool.transforms import sample_points
from pcam.tool.pointcloud import make_open3d_point_cloud

from pcam.tool.comp import metric_func_bop_for_others, metric_func_modelnet_for_others
# import sys
# sys.path.append('/home/hush/projects/oprnet-spos')
# from oprnet_utils.comp import metric_func_bop_for_others

def ortho2rotation(poses):
    """
    Function copy-pasted from https://github.com/chrischoy/DeepGlobalRegistration/blob/master.
    Needed for DGR's refinement [1-3]
    """
    r"""
    poses: batch x 6
    """
    def normalize_vector(v):
        r"""
        Batch x 3
        """
        v_mag = torch.sqrt((v**2).sum(1, keepdim=True))
        v_mag = torch.clamp(v_mag, min=1e-8)
        v = v / v_mag
        return v

    def cross_product(u, v):
        r"""
        u: batch x 3
        v: batch x 3
        """
        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        i = i[:, None]
        j = j[:, None]
        k = k[:, None]
        return torch.cat((i, j, k), 1)

    def proj_u2a(u, a):
        r"""
        u: batch x 3
        a: batch x 3
        """
        inner_prod = (u * a).sum(1, keepdim=True)
        norm2 = (u**2).sum(1, keepdim=True)
        norm2 = torch.clamp(norm2, min=1e-8)
        factor = inner_prod / norm2
        return factor * u

    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - proj_u2a(x, y_raw))
    z = cross_product(x, y)

    x = x[:, :, None]
    y = y[:, :, None]
    z = z[:, :, None]
    return torch.cat((x, y, z), 2)


class Transformation(torch.nn.Module):
    """
    Function copy-pasted from https://github.com/chrischoy/DeepGlobalRegistration/blob/master.
    Needed for DGR's refinement [1-3]
    """
    def __init__(self, R_init=None, t_init=None):
        torch.nn.Module.__init__(self)
        rot_init = torch.rand(1, 6)
        trans_init = torch.zeros(1, 3)
        if R_init is not None:
          rot_init[0, :3] = R_init[:, 0]
          rot_init[0, 3:] = R_init[:, 1]
        if t_init is not None:
          trans_init[0] = t_init

        self.rot6d = torch.nn.Parameter(rot_init)
        self.trans = torch.nn.Parameter(trans_init)

    def forward(self, points):
        rot_mat = ortho2rotation(self.rot6d)
        return points @ rot_mat[0].t() + self.trans


class HighDimSmoothL1Loss:
    """
    Function copy-pasted from https://github.com/chrischoy/DeepGlobalRegistration/blob/master.
    Needed for DGR's refinement [1-3]
    """
    def __init__(self, weights, quantization_size=1, eps=np.finfo(np.float32).eps):
        self.eps = eps
        self.quantization_size = quantization_size
        self.weights = weights
        if self.weights is not None:
            self.w1 = weights.sum()
            self.w1 = 1e-8 if self.w1 == 0 else self.w1

    def __call__(self, X, Y):
        sq_dist = torch.sum(((X - Y) / self.quantization_size)**2, axis=1, keepdim=True)
        use_sq_half = 0.5 * (sq_dist < 1).float()

        loss = (0.5 - use_sq_half) * (torch.sqrt(sq_dist + self.eps) -
                                      0.5) + use_sq_half * sq_dist

        if self.weights is None:
            return loss.mean()
        else:
            return (loss * self.weights).sum() / self.w1


def test(test_loader, net, device,
         threshold_rte, threshold_rre,
         dataset, num_points=None, voxel_size=None, 
         icp=False,
         dgr_optim=False,
         safeguard=False, nb_average=0, 
         wsum_threshold=0):
    # Init.
    times = []
    if dataset == "3dmatch" or dataset == "kitti":
        logger = Logger()
        # To get 3D match scene name
        subset_names = open(test_loader.dataset.DATA_FILES[test_loader.dataset.phase]).read().split()
        print(subset_names)
        array_sid = []
    elif "modelnet" in dataset:
        pred_rs = []
        pred_ts = []
        true_rs = []
        true_ts = []
    net = net.eval()
    search = lcp_nn.SearchQuantized(K=32, stride=1)

    #
    for it, batch in enumerate(tqdm(test_loader)):

        # --- Start timer
        torch.cuda.synchronize()
        start = time.perf_counter()

        # --- Extract data
        # Get full point cloud
        src = batch['raw_src']
        tgt = batch['raw_tgt']
        assert src.shape[0] == 1
        # Quantize
        if voxel_size is not None:
            sel0 = ME.utils.sparse_quantize(src[0] / voxel_size, return_index=True)
            sel1 = ME.utils.sparse_quantize(tgt[0] / voxel_size, return_index=True)
            src = src[0][sel0]
            tgt = tgt[0][sel1]
            src_icp = src
            tgt_icp = tgt

        #
        cat_src, cat_tgt = [], []
        cat_ind_src, cat_ind_tgt = [], []
        for ind_average in range(nb_average):
            # Sample
            if num_points is not None:
                src = torch.Tensor(sample_points(src, num_points).T).unsqueeze(0)
                tgt = torch.Tensor(sample_points(tgt, num_points).T).unsqueeze(0)
            # Search neighbors
            src_indices = search(src)[0].to(device)
            tgt_indices = search(tgt)[0].to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            #
            cat_src.append(src)
            cat_tgt.append(tgt)
            cat_ind_src.append(src_indices)
            cat_ind_tgt.append(tgt_indices)
        src = torch.cat(cat_src, 0)
        tgt = torch.cat(cat_tgt, 0)
        src_indices = torch.cat(cat_ind_src, 0)
        tgt_indices = torch.cat(cat_ind_tgt, 0)

        # Match points
        with torch.no_grad():
            Rest, Test, corres_pts_for_pts1, _, _, _, w_pts1, _ = \
                net(src, tgt, indices1=src_indices, indices2=tgt_indices)

        #
        error = torch.norm(torch.bmm(Rest, src) + Test - corres_pts_for_pts1, p=2, dim=1, keepdim=True) ** 2
        error = (w_pts1 * error).mean(-1)
        ind = torch.argmin(error, dim=0)
        Rest = Rest[ind:ind+1]
        Test = Test[ind:ind+1]

        # DGR safeguard
        wsum = w_pts1.sum().item()
        if safeguard and wsum < wsum_threshold:
            """Safeguard as proposed by DGR"""
            pcd0 = make_open3d_point_cloud(src[0].T.cpu().numpy())
            pcd1 = make_open3d_point_cloud(corres_pts_for_pts1[0].T.cpu().numpy())
            idx0 = np.arange(src.shape[-1]).astype('int')
            idx1 = np.arange(corres_pts_for_pts1.shape[-1]).astype('int')
            corres = np.stack((idx0, idx1), axis=1)
            corres = o3d.utility.Vector2iVector(corres)
            #
            T = o3d.registration.registration_ransac_based_on_correspondence(
                pcd0,
                pcd1,
                corres,
                2 * voxel_size,
                o3d.registration.TransformationEstimationPointToPoint(False),
                4,
                o3d.registration.RANSACConvergenceCriteria(4000000, 80000)
            ).transformation
            #
            Rest = torch.from_numpy(T[0:3, 0:3]).unsqueeze(0).float().to(device)
            Test = torch.from_numpy(T[0:3, 3]).unsqueeze(0).unsqueeze(-1).float().to(device)

        else:
            if dgr_optim:
                """Optional refinement by optimisation as used in DGR"""
                len_src = src_icp.shape[0]
                src = torch.cat([src[i:i + 1] for i in range(nb_average)], -1)[:, :, :len_src]
                w_pts1 = torch.cat([w_pts1[i:i+1] for i in range(nb_average)], -1)[:, :, :len_src]
                corres_pts_for_pts1 = torch.cat([corres_pts_for_pts1[i:i + 1] for i in range(nb_average)], -1)[:, :, :len_src]

                max_break_count = 20
                break_threshold_ratio = 1e-4
                loss_fn = HighDimSmoothL1Loss(w_pts1.transpose(1, 2), 2 * voxel_size)
                transformation = Transformation(Rest[0], Test[0, :, 0]).to(device)

                optimizer = optim.Adam(transformation.parameters(), lr=1e-1)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
                loss_prev = loss_fn(transformation(src[0].t()), corres_pts_for_pts1[0].t()).item()
                break_counter = 0

                # Transform points
                for _ in range(1000):
                    new_points = transformation(src[0].t())
                    loss = loss_fn(new_points, corres_pts_for_pts1[0].t())
                    if loss.item() < 1e-7:
                        break
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if abs(loss_prev - loss.item()) < loss_prev * break_threshold_ratio:
                      break_counter += 1
                      if break_counter >= max_break_count:
                        break
                    loss_prev = loss.item()
                rot6d = transformation.rot6d.detach()
                trans = transformation.trans.detach()
                Rest = ortho2rotation(rot6d)
                Test = trans.unsqueeze(-1)

        # ICP (on non subsampled but quantized point cloud)
        if icp:
            """Optional refinement by ICP"""
            assert Rest.shape[0] == 1
            T = np.identity(4)
            T[0:3, 0:3] = Rest[0].cpu().numpy()
            T[0:3, 3] = Test[0, :, 0].cpu().numpy()
            
            T = o3d.registration.registration_icp(
                make_open3d_point_cloud(src_icp),
                make_open3d_point_cloud(tgt_icp),
                icp,
                T,
                o3d.registration.TransformationEstimationPointToPoint()
            ).transformation
            Rest = torch.from_numpy(T[0:3, 0:3]).unsqueeze(0).float().to(device)
            Test = torch.from_numpy(T[0:3, 3]).unsqueeze(0).unsqueeze(-1).float().to(device)

        # End timer
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

        # Save metric
        R = batch['rotation'].float().to(device)
        T = batch['translation'].float().to(device)
        # Compensate for centering in dataloader (correct bias at training)
        p0_mean = batch['p0_mean'].float().to(device)
        p1_mean = batch['p1_mean'].float().to(device)
        Test = Test + p1_mean - Rest @ p0_mean
        if "modelnet" not in dataset:
            if batch['filename'][0] in subset_names:
                array_sid.append(subset_names.index(batch['filename'][0]))
            save_metrics(logger, 'val', R, T, Rest, Test, threshold_rte, threshold_rre)
        elif "modelnet" in dataset:
            pred_rs.append(Rest.detach().cpu().numpy())
            pred_ts.append(Test.squeeze().detach().cpu().numpy())
            true_rs.append(R.detach().cpu().numpy())
            true_ts.append(T.squeeze().detach().cpu().numpy())

    # Log
    if "modelnet" not in dataset:
        print("RTE all:", np.mean(logger.store['val.rte_all']))
        print("RRE all", np.mean(logger.store['val.rre_all']))
        print("Recall:", np.mean(logger.store['val.recall']))
        print("RTE: ", np.mean(logger.store['val.rte']))
        print("RRE: ", np.mean(logger.store['val.rre']))
        print("Times:", np.mean(times))
        return np.array(logger.store['val.rte_all']), np.array(logger.store['val.rre_all']), np.array(times), np.array(array_sid), subset_names
    elif "modelnet" in dataset:
        pred_r = np.concatenate(pred_rs, axis=0)
        pred_t = np.concatenate(pred_ts, axis=0)
        true_r = np.concatenate(true_rs, axis=0)
        true_t = np.concatenate(true_ts, axis=0)
        print(pred_r.shape, true_r.shape, pred_t.shape, true_t.shape)
        metrics = compute_metrics(true_r, pred_r, true_t, pred_t)
        print(metrics)


def test_bop(test_loader, net, device,
         threshold_rte, threshold_rre,
         dataset_name, num_points=None, voxel_size=None, 
         icp=False,
         dgr_optim=False,
         safeguard=False, nb_average=0, 
         wsum_threshold=0):
    net = net.eval()
    search = lcp_nn.SearchQuantized(K=32, stride=1)

    from nni.nas.pytorch.utils import AverageMeterGroup
    meters = AverageMeterGroup()
    import pandas as pd
    results = pd.DataFrame(columns=('scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'))

    srcs = []
    tgts = []
    R_gts = []
    t_gts = []
    r_pred = []
    t_pred = []
    model_scales = []
    diameters = []

    scene_ids = []
    im_ids = []
    obj_ids = []
    inliers_count = 0

    #
    for it, batch in enumerate(tqdm(test_loader)):

        # --- Extract data
        # Get full point cloud
        src, tgt, _, _, _, _ = batch
        assert src.shape[0] == 1
        # Quantize
        if voxel_size is not None:
            print('voxel_size is not None')
        #     sel0 = ME.utils.sparse_quantize(src[0] / voxel_size, return_index=True)
        #     sel1 = ME.utils.sparse_quantize(tgt[0] / voxel_size, return_index=True)
        #     src = src[0][sel0]
        #     tgt = tgt[0][sel1]
        #     src_icp = src
        #     tgt_icp = tgt

        #
        cat_src, cat_tgt = [], []
        cat_ind_src, cat_ind_tgt = [], []
        for ind_average in range(nb_average): # 1
            # Sample
            if num_points is not None:
                print('num_points is not None')
            #     src = torch.Tensor(sample_points(src, num_points).T).unsqueeze(0)
            #     tgt = torch.Tensor(sample_points(tgt, num_points).T).unsqueeze(0)
            # Search neighbors

            src_indices = search(src)[0].to(device)
            tgt_indices = search(tgt)[0].to(device)
            src = src.to(device)
            tgt = tgt.to(device)
            #
            cat_src.append(src)
            cat_tgt.append(tgt)
            cat_ind_src.append(src_indices)
            cat_ind_tgt.append(tgt_indices)
        src = torch.cat(cat_src, 0)
        tgt = torch.cat(cat_tgt, 0)
        src_indices = torch.cat(cat_ind_src, 0)
        tgt_indices = torch.cat(cat_ind_tgt, 0)

        # Match points
        with torch.no_grad():
            Rest, Test, corres_pts_for_pts1, _, _, _, w_pts1, _ = \
                net(src, tgt, indices1=src_indices, indices2=tgt_indices)
        #
        error = torch.norm(torch.bmm(Rest, src) + Test - corres_pts_for_pts1, p=2, dim=1, keepdim=True) ** 2
        error = (w_pts1 * error).mean(-1)
        ind = torch.argmin(error, dim=0)
        Rest = Rest[ind:ind+1]
        Test = Test[ind:ind+1]

        # Save metric
        r_pred.append(Rest.detach().cpu().numpy())
        t_pred.append(Test.detach().cpu().numpy())
        src_ori, tgt_ori, R_gt, t_gt, _, info_dict = batch
        scene_ids.append(info_dict['scene_id'].numpy().squeeze())
        im_ids.append(info_dict['im_id'].numpy().squeeze())
        obj_ids.append(info_dict['obj_id'].numpy().squeeze())
        srcs.append(src_ori.numpy())
        tgts.append(tgt_ori.numpy())
        R_gts.append(R_gt.numpy())
        t_gts.append(t_gt.numpy())
        model_scales.append(info_dict['model_scale'].numpy().squeeze())
        diameters.append(info_dict['diameter'].numpy())


        target = corres_pts_for_pts1.detach().cpu()
        x = (target - (R_gt@src.detach().cpu() + t_gt.reshape(1,3,1)))**2
        dist = torch.sqrt(x.squeeze().sum(0))
        dist = dist.numpy()
        inlier_true = (dist < 0.05).sum()
        inliers_count += inlier_true


    R_gts, t_gts, r_pred, t_pred = np.concatenate(R_gts, axis=0), np.concatenate(t_gts, axis=0), np.concatenate(r_pred, axis=0), np.concatenate(t_pred, axis=0)
    srcs, tgts = np.concatenate(srcs, axis=0), np.concatenate(tgts, axis=0)
    
    print(srcs.shape[0])
    print(inliers_count / srcs.shape[0])
    
    # for i, (scene_id, im_id, obj_id, r, t, model_scale) in enumerate(zip(scene_ids, im_ids, obj_ids, r_pred, t_pred, model_scales)):
    for i, (scene_id, im_id, obj_id, r, t, model_scale, src, tgt) in enumerate(zip(scene_ids, im_ids, obj_ids, r_pred, t_pred, model_scales, srcs, tgts)):
                
        src, tgt, t = src/model_scale, tgt/model_scale, t/model_scale
        # With ICP
        A = np.eye(4)
        A[:3, :3] = r
        A[:3, 3] = t.squeeze()
        r_icp, t_icp = icp_func(src, tgt, A)
        r, t = r_icp.numpy(), t_icp.numpy()
        

        r = r.reshape(1,-1).squeeze().astype(np.str).tolist()
        t = t.squeeze().astype(np.str).tolist()
        r_str = ' '.join(r)
        t_str = ' '.join(t)
        result = {'scene_id': scene_id, 'im_id': im_id, 'obj_id': obj_id,
            'score': 1, 'R': r_str, 't': t_str, 'time': -1}
        results.loc[results.shape[0]+1] = result
                
    # results.to_csv('0325pcamvox_{}-test.csv'.format(dataset_name), index = False, header = True)

    metrics = metric_func_modelnet_for_others(srcs, tgts, R_gts, t_gts, r_pred, t_pred)
    # metrics = metric_func_bop_for_others(srcs, tgts, R_gts, t_gts, r_pred, t_pred, model_scales, diameters)
    meters.update(metrics)
    from termcolor import colored
    import logging
    logger = logging.getLogger("pcam")
    logger.info(colored(f'\nvalidation'
            f'\nmAP_R:[{meters.mAP_R5.avg:.2f}, {meters.mAP_R10.avg:.2f}, {meters.mAP_R15.avg:.2f}, '
            f'{meters.mAP_R20.avg:.2f}, {meters.mAP_R25.avg:.2f}, {meters.mAP_R30.avg:.2f}] '
            f'(avg:{meters.mAP_R_mean.avg:.2f}) '
            f'\nmAP_t:[{meters.mAP_t5.avg:.2f}, {meters.mAP_t10.avg:.2f}, {meters.mAP_t15.avg:.2f}, '
            f'{meters.mAP_t20.avg:.2f}, {meters.mAP_t25.avg:.2f}, {meters.mAP_t30.avg:.2f}] '
            f'(avg:{meters.mAP_t_mean.avg:.2f}) '
            f'\nADD:[{meters.add.avg:.2f}] ADDS:[{meters.adds.avg:.2f}]', 'green' )
            )


def icp_func(src, tgt, current_transformation=None, method='point2point'):
    # random initialize the rotation, using the translation of two point cloud sets' center as initionzation.
    src_o3d, tgt_o3d = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    src_o3d.points = o3d.utility.Vector3dVector(src.T)
    tgt_o3d.points = o3d.utility.Vector3dVector(tgt.T)
    if method == 'point2point':
        result = o3d.pipelines.registration.registration_icp(
                    src_o3d, tgt_o3d, 20., current_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()) 
        
    elif method == 'point2plane':
        radius = 0.05
        max_nn = 100
        src_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        tgt_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        result = o3d.pipelines.registration.registration_icp(
                    src_o3d, tgt_o3d, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    trans = torch.from_numpy(np.array(result.transformation)).float()
    return trans[:3, :3], trans[:3, 3][:, None]