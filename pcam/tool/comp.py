import numpy as np
import torch
from scipy.spatial.transform import Rotation
from sklearn.metrics import mean_squared_error
# from oprnet_utils.visual import plot_match, plot_point_cloud
import os
import open3d as o3d
# for bop vsd error metric
# from bop_toolkit_lib import pose_error, renderer
# from bop_toolkit_lib import visualization

import sklearn.metrics as metrics
# from oprnet_utils.rpmnet_funcs import compute_rigid_transform
# from oprnet_utils.pytorch3d_utils import build_bijective_label

def calc_pose(src, tgt):
    reflect = torch.eye(3)
    reflect[2, 2] = -1
    reflect = reflect.to(src)
    src_centered = src - src.mean(dim=1, keepdim=True)
    tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)

    H = torch.matmul(src_centered, tgt_centered.transpose(1, 0).contiguous())
    u, s, v = torch.svd(H)
    r = torch.matmul(v, u.transpose(1, 0).contiguous())
    r_det = torch.det(r)
    if r_det < 0:
        v = torch.matmul(v, reflect)
        r = torch.matmul(v, u.transpose(1, 0).contiguous())
    t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)
    return r, t

def degree_err(R_gt, R_pred, eps=1e-16):
    err = np.arccos(
      np.clip((np.trace(R_pred[:3, :3].T @ R_gt[:3, :3]) - 1) / 2, -1 + eps,
              1 - eps)) * 180. / np.pi
    return err

def mse_period(R_gt, R_pred):
    # remove the period issue.
    mse_0 = ((R_gt - R_pred)**2)
    mse_1 = ((R_gt - (R_pred + 360))**2)
    mse_2 = ((R_gt - (R_pred - 360))**2)
    mse = np.stack([mse_0, mse_1, mse_2])
    mse = mse.min(0).mean()
    return mse

def mae_period(R_gt, R_pred):
    # remove the period issue.
    mae_0 = np.absolute(R_gt - R_pred)
    mae_1 = np.absolute(R_gt - (R_pred + 360))
    mae_2 = np.absolute(R_gt - (R_pred - 360))
    mae = np.stack([mae_0, mae_1, mae_2])
    mae = mae.min(0).mean()
    return mae

def trans_err(t_gt, t_pred):
    return torch.sqrt(torch.sum((t_gt - t_pred)**2))

def mAP_th(err, bins=[0., 5., 10., 15., 20., 25., 30.]):
    '''
    different threshold, summarize it and draw the pic.
    '''
    total_num = len(err)
    hist, _ = np.histogram(err, bins=bins)
    mAP = [np.sum(hist[:i+1]) for i in range(hist.shape[0])]
    mAP = np.array(mAP) / total_num
    return mAP

def count_file(file_path):
    file_list = os.listdir(file_path)
    return len(file_list)


def metric_func_bop_save(source, target, scores, info_dict):
    scene_ids = []
    im_ids = []
    obj_ids = []
    Rs = []
    ts = []

    for _, (score, src, tgt, model_scale, scene_id, im_id, obj_id) in enumerate(zip(scores.cpu(), source.cpu(), target.cpu(), \
                        info_dict['model_scale'], info_dict['scene_id'], info_dict['im_id'], info_dict['obj_id'])):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)

        t /= model_scale

        scene_ids.append(scene_id.numpy())
        im_ids.append(im_id.numpy())
        obj_ids.append(obj_id.numpy())
        Rs.append(r.numpy())
        ts.append(t.numpy())


        # print(mse_period(R, r))
        # print(mean_squared_error(T, t))
        # A = np.eye(4)
        # A[:3, :3] = R_
        # A[:3, 3] = T_.squeeze()
        # file_id = count_file('./plot_fig') + 1
        # plot_point_cloud(src.T, tgt.T, A, f'plot_fig/img_{file_id}.gif')


    # scene_ids = np.concatenate(scene_ids, axis=0)
    # im_ids = np.concatenate(im_ids, axis=0)
    # obj_ids = np.concatenate(obj_ids, axis=0)
    # Rs = np.concatenate(Rs, axis=0)
    # ts = np.concatenate(ts, axis=0)


    return scene_ids, im_ids, obj_ids, Rs, ts


def metric_func_bop_for_others(srcs, targets, rotations, translations, rotation_preds, translation_preds, model_scales, diameters):
    trans_d = []
    degree_r = []
    add_list = []
    
    trans_th = [0., .5, 1., 2., 5., 10., 15.] # bop_dataset in mm
    trans_th = [item * 10. for item in trans_th] # in cm

    for _, (src, tgt, R, T, r, t, model_scale, diameter) in enumerate(zip(srcs, targets, rotations, translations, rotation_preds, translation_preds, model_scales, diameters)):
        
        R_, T_, r_, t_ = R, T, r, t

        # tmp = Rotation.from_matrix(R)
        tmp = Rotation.from_matrix(R)
        R = tmp.as_euler('zyx', degrees=True)
        # tmp = Rotation.from_matrix(r)
        tmp = Rotation.from_matrix(r)
        r = tmp.as_euler('zyx', degrees=True)

        src /= model_scale
        T /= model_scale
        t /= model_scale

        # icp
        A = np.eye(4)
        A[:3, :3] = r_
        A[:3, 3] = t.squeeze()
        tgt /= model_scale
        r_icp, t_icp = icp(src, tgt, A)
        r_, t = r_icp.numpy(), t_icp.numpy()

        # A = np.eye(4)
        # A[:3, :3] = r_
        # A[:3, 3] = t_.squeeze()
        # # file_id = count_file('./plot_fig') + 1
        # # plot_point_cloud(src.T, tgt.T, A, f'plot_fig/img_{file_id}.gif')
        # plot_point_cloud(src.T, tgt.T, A, f'plot.gif')
        try:
            degree_r.append(degree_err(R_, r_))
            T, t = T.squeeze(), t.squeeze()
            trans_d.append(trans_err(torch.from_numpy(T), torch.from_numpy(t)))

            # A = np.eye(4)
            # A[:3, :3] = r_
            # A[:3, 3] = t_.squeeze()
            # file_id = count_file('./plot_res') + 1
            # plot_point_cloud(src.T, tgt.T, A, name=f'./plot_res/{file_id}.gif')
            # plot_match(src.T.numpy(), tgt.T.numpy(), A, score.numpy())
            # import ipdb; ipdb.set_trace()

            # mse_r.append(mse_period(R, r))
            # mse_t.append(mean_squared_error(T, t))
            # mae_r.append(mae_period(R, r))
            # mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            # mse_r.append(360.)
            # mse_t.append(360.)
            # mae_r.append(360.)
            # mae_t.append(360.)

        # ADD 
        if T.shape[0] != 3: T = T.T
        if t.shape[0] != 3: t = t.T
        try:
            src_gt = R_ @ src + T
            src_eval  = r_ @ src + t
        except:
            src_gt = R_ @ src + T[:, np.newaxis]
            src_eval  = r_ @ src + t[:, np.newaxis]
            # print('trans dim err')

        add = torch.sum((torch.from_numpy(src_gt) - torch.from_numpy(src_eval))**2, 0).sqrt().mean()
        th = .1 * diameter
        add_list.append(add.numpy() < th)

    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(trans_d, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(trans_d, trans_th).mean(),
        'add': np.asarray(add_list).sum()/len(add_list),
        'adds': 1234,
    }
    return metrics

def icp(src, tgt, current_transformation=None, method='point2point'):
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
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        result = o3d.pipelines.registration.registration_icp(
                    source, target, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    trans = torch.from_numpy(np.array(result.transformation)).float()
    return trans[:3, :3], trans[:3, 3][:, None]

def metric_func_modelnet_for_others(srcs, targets, rotations, translations, rotation_preds, translation_preds):
    mse_r, mse_t = [], []
    mae_r, mae_t = [], []
    degree_r = []
    trans_th = [0., 1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1, 5.e-1] # modelnet40

    for _, (src, target, R, T, r, t) in enumerate(zip(srcs, targets, rotations, translations, rotation_preds, translation_preds)):
        
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R)
        R = tmp.as_euler('zyx', degrees=True)
        tmp = Rotation.from_matrix(r)
        r = tmp.as_euler('zyx', degrees=True)

        # src /= model_scale
        # T /= model_scale
        # t /= model_scale
        # target /= model_scale

        # A = np.eye(4)
        # A[:3, :3] = r_
        # A[:3, 3] = t_.squeeze()
        # file_id = count_file('./plot_fig') + 1
        # plot_point_cloud(src.T, target.T, A, f'plot_fig/img_{file_id}.gif')


        try:
            degree_r.append(degree_err(R_, r_))

            # if degree_r[-1] < 5:
            #     A = np.eye(4)
            #     A[:3, :3] = r_
            #     A[:3, 3] = t_.squeeze()
            #     file_id = count_file('./plot_res') + 1
            #     plot_point_cloud(src.T, tgt.T, A)
            #     # plot_point_cloud(src.T, tgt.T, A, name=f'./plot_res/{file_id}.gif')
            #     # plot_match(src.T.numpy(), tgt.T.numpy(), A, score.numpy())
            #     import ipdb; ipdb.set_trace()

            mse_r.append(mse_period(R, r))
            mse_t.append(mean_squared_error(T, t))
            mae_r.append(mae_period(R, r))
            mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            mse_r.append(360.)
            mse_t.append(360.)
            mae_r.append(360.)
            mae_t.append(360.)


    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(mse_t, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(mse_t, trans_th).mean(),
        'add': 1234,
        'adds': 1234,
    }
    return metrics


def metric_func_bop(source, target, rotation, translation, scores, info_dict):
    trans_d = []
    degree_r = []
    add_list = []
    
    trans_th = [0., .5, 1., 2., 5., 10., 15.] # bop_dataset in mm
    trans_th = [item * 10. for item in trans_th] # in cm

    for _, (score, src, tgt, R, T, model_scale, diameter) in enumerate(zip(scores.cpu(), source.cpu(), target.cpu(), rotation.cpu(), translation.cpu(), info_dict['model_scale'], info_dict['diameter'])):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])

        mask1 = (row != score.shape[0] - 1).type(torch.bool)
        mask2 = (val > 0.02).type(torch.bool)
        mask = mask1 & mask2

        # mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R.numpy())
        R = tmp.as_euler('zyx', degrees=True)
        tmp = Rotation.from_matrix(r.numpy())
        r = tmp.as_euler('zyx', degrees=True)

        # print(src.max())
        # print(src.min())
        # print(tgt.max())
        # print(tgt.min())

        src /= model_scale
        T /= model_scale
        t /= model_scale

        try:
            degree_r.append(degree_err(R_, r_))
            trans_d.append(trans_err(T, t))
            # if degree_r[-1] < 5:
            # A = np.eye(4)
            # A[:3, :3] = R_
            # A[:3, 3] = T_.squeeze()
            # file_id = count_file('./plot_fig') + 1
            # plot_point_cloud(src.T, tgt.T, A, name=f'./plot_fig/{file_id}.gif')
            # print('1')
            # #     # plot_match(src.T.numpy(), tgt.T.numpy(), A, score.numpy())
            #     import ipdb; ipdb.set_trace()

            # mse_r.append(mse_period(R, r))
            # mse_t.append(mean_squared_error(T, t))
            # mae_r.append(mae_period(R, r))
            # mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            # mse_r.append(360.)
            # mse_t.append(360.)
            # mae_r.append(360.)
            # mae_t.append(360.)

        # ADD 
        src_gt = R_ @ src + T
        src_eval  = r_ @ src + t
        add = torch.sum((src_gt - src_eval)**2, 0).sqrt().mean()
        th = .1 * diameter
        add_list.append(add < th)

    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(trans_d, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(trans_d, trans_th).mean(),
        'add': np.asarray(add_list).sum()/len(add_list),
        'adds': 1234, # notice this is just a placeholder.
    }
    return metrics

def metric_func_modelnet(source, target, rotation, translation, scores, info_dict, bi_layer):
    mse_r, mse_t = [], []
    mae_r, mae_t = [], []
    trans_d = []
    add_list = []
    degree_r = []
    trans_th = [0., 1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1, 5.e-1] # modelnet40


    for _, (score, src, tgt, R, T, _, _) in enumerate(zip(scores.cpu(), source.cpu(), target.cpu(), rotation.cpu(), translation.cpu(), info_dict['model_scale'], info_dict['diameter'])):
        
        if bi_layer == 'prnet':
            # val, col = torch.max(score, 1)
            # row = torch.arange(score.shape[0])
            # mask = (col != score.shape[1]).type(torch.bool) 
            # src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
            # r, t = calc_pose(src_m, tgt_m)
            # val, row = torch.max(score, 0)

            # col = torch.arange(score.shape[1])
            # # mask = (row != score.shape[0]).type(torch.bool) 
            # mask1 = (row != score.shape[0]).type(torch.bool)
            # mask2 = (val > 0.3).type(torch.bool)
            # mask = mask1 & mask2
            # src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
            # r, t = calc_pose(src_m, tgt_m)

            src_corr = torch.matmul(tgt, score.transpose(1, 0).contiguous())

            src_centered = src - src.mean(dim=1, keepdim=True)

            src_corr_centered = src_corr - src_corr.mean(dim=1, keepdim=True)

            H = torch.matmul(src_centered, src_corr_centered.transpose(1, 0).contiguous()).cpu()
            try:
                u, s, v = torch.svd(H)
                r = torch.matmul(v, u.transpose(1, 0)).contiguous()
                r_det = torch.det(r).item()
                diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                                [0, 1.0, 0],
                                                [0, 0, r_det]]).astype('float32'))
                r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
                t = torch.matmul(-r, src.mean(dim=1, keepdim=True)) + src_corr.mean(dim=1, keepdim=True)

                R_, T_, r_, t_ = R, T, r, t

                tmp = Rotation.from_matrix(R.numpy())
                R = tmp.as_euler('zyx', degrees=True)
                tmp = Rotation.from_matrix(r.numpy())
                r = tmp.as_euler('zyx', degrees=True)
            except:
                print('svd err.')
                file_id = count_file('./plot_fig') + 1
                torch.save(H, f'plot_fig/H_{file_id}.pth')


        elif bi_layer == 'softmax':
            # import ipdb
            # ipdb.set_trace()

            val, row = torch.max(score, 0)
            col = torch.arange(score.shape[1])
            mask = (row != score.shape[0]).type(torch.bool) 
            src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
            r, t = calc_pose(src_m, tgt_m)
            

            # tgt_corr = torch.matmul(src, score)
            # tgt_centered = tgt - tgt.mean(dim=1, keepdim=True)
            # tgt_corr_centered = tgt_corr - tgt_corr.mean(dim=1, keepdim=True)
            # h = torch.matmul(tgt_corr_centered, tgt_centered.transpose(1, 0).contiguous())
            # reflect = torch.eye(3)
            # reflect[2, 2] = -1

            # u, s, v = torch.svd(h)
            # r = torch.matmul(v, u.transpose(1, 0).contiguous())
            # r_det = torch.det(r)
            # if r_det < 0:
            #     u, s, v = torch.svd(h)
            #     v = torch.matmul(v, reflect)
            #     r = torch.matmul(v, u.transpose(1, 0).contiguous())
            #     # r = r * self.reflect
            # t = torch.matmul(-r, tgt_corr.mean(dim=1, keepdim=True)) + tgt.mean(dim=1, keepdim=True)

        elif bi_layer == 'rpmnet':
            weighted_tgt = score @ tgt.T / (torch.sum(score, dim=1, keepdim=True) + 1e-5)
            src = src.T
            transform = compute_rigid_transform(src.unsqueeze(0), weighted_tgt.unsqueeze(0), weights=torch.sum(score, dim=1).unsqueeze(0))
            r = transform[0, :3, :3]
            t = transform[0, :3, 3]

        elif bi_layer == 'ot':
            score[-1, -1] = np.inf
            val, row = torch.max(score, 0)
            col = torch.arange(score.shape[1])

            # mask1 = (row != score.shape[0] - 1).type(torch.bool)
            # mask2 = (val > 0.3).type(torch.bool)
            # mask = mask1 & mask2
            mask = (row != score.shape[0] - 1).type(torch.bool)
            src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
            r, t = calc_pose(src_m, tgt_m)

        else:
            raise('args.bi_layer type error')

        if bi_layer != 'prnet':
            R_, T_, r_, t_ = R, T, r, t

            tmp = Rotation.from_matrix(R.numpy())
            R = tmp.as_euler('zyx', degrees=True)
            tmp = Rotation.from_matrix(r.numpy())
            r = tmp.as_euler('zyx', degrees=True)

        try:
            degree_r.append(degree_err(R_, r_))
            # trans_d.append(trans_err(T, t))
            # if degree_r[-1] < 5:
            # A = np.eye(4)
            # A[:3, :3] = R_
            # A[:3, 3] = T_.squeeze()
            # file_id = count_file('./plot_fig') + 1
            # plot_point_cloud(src.T, tgt.T, A, name=f'./plot_fig/{file_id}.gif')
                # plot_point_cloud(src.T, tgt.T, A, name=f'./plot_fig/{file_id}.gif')
                # plot_match(src.T.numpy(), tgt.T.numpy(), A, score.numpy())
            #     import ipdb; ipdb.set_trace()

            mse_r.append(mse_period(R, r))
            mse_t.append(mean_squared_error(T, t))
            mae_r.append(mae_period(R, r))
            mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            mse_r.append(360.)
            mse_t.append(360.)
            mae_r.append(360.)
            mae_t.append(360.)


    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(mse_t, trans_th)
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(mse_t, trans_th).mean(),
        'add': 1234,
        'adds': 1234, # notice this is just a placeholder.
    }
    return metrics


def plot_pics(source, target, rotation, translation, scores, labels):

    mse_r, mae_r = [], []
    mse_t, mae_t = [], []
    degree_r = []
    for _, (score, src, tgt, R, T, label) in enumerate(zip(scores.cpu(), source.cpu(), target.cpu(), rotation.cpu(), translation.cpu(), labels.cpu())):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R.numpy())
        R = tmp.as_euler('zyx', degrees=True)
        tmp = Rotation.from_matrix(r.numpy())
        r = tmp.as_euler('zyx', degrees=True)

        print(degree_err(R_, r_))
        print(R)
        print(r)
        print(T.numpy().reshape(1,3))
        print(t.numpy().reshape(1,3))

        A = np.eye(4)
        # cal pose
        A[:3, :3] = r_
        A[:3, 3] = t_.squeeze()
        # # gt pose
        # A[:3, :3] = R_
        # A[:3, 3] = T_.squeeze()
        file_id = count_file('./plot_fig') + 1
        # plot_point_cloud(src.T, tgt.T, A, f'plot_fig/img_{file_id}.gif')
        plot_match(src.numpy().T, tgt.numpy().T, A, label.numpy(), name='plot_fig/match_%d.gif'%file_id)

        try:
            degree_r.append(degree_err(R_, r_))
            # if degree_r[-1] < 15:
            #     A = np.eye(4)
            #     A[:3, :3] = r_
            #     A[:3, 3] = t_.squeeze()
            #     file_id = count_file('./plot_fig') + 1
            #     plot_point_cloud(src.T, tgt.T, A, f'plot_fig/img_{file_id}.gif')
            mse_r.append(mse_period(R, r))
            mse_t.append(mean_squared_error(T, t))
            mae_r.append(mae_period(R, r))
            mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            mse_r.append(360.)
            mse_t.append(360.)
            mae_r.append(360.)
            mae_t.append(360.)
    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(mse_t, [0., 1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1, 5.e-1])
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(mse_t, [0., 1.e-3, 5.e-3, 1.e-2]).mean(),
        'add': 1234, # notice this is just a placeholder.
        'adds': 1234,
    }
    return metrics

def average_distance(source, target, rotation, translation, scores, info_dict):
    '''
    # this is only for ycbv
    According to the paper: Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes
    '''
    add_list = []
    add_s_list = []
    mse_r, mae_r = [], []
    mse_t, mae_t = [], []
    degree_r = []
    for _, (score, src, tgt, R, T, model_scale, diameter) in enumerate(zip(scores.cpu(), 
                                                    source.cpu(), 
                                                    target.cpu(), 
                                                    rotation.cpu(), 
                                                    translation.cpu(),
                                                    info_dict['model_scale'],
                                                    info_dict['diameter'])):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R.numpy())
        R = tmp.as_euler('zyx', degrees=True)
        tmp = Rotation.from_matrix(r.numpy())
        r = tmp.as_euler('zyx', degrees=True)

        # from oprnet_utils.visual import plot_point_cloud
        # A = np.eye(4)
        # A[:3, :3] = r_
        # A[:3, 3] = t_.squeeze()
        # plot_point_cloud(src.T, tgt.T, A)
        # import ipdb; ipdb.set_trace()

        # transfer back to the original size
        src /= model_scale
        T /= model_scale
        t /= model_scale

        try:
            degree_r.append(degree_err(R_, r_))
            mse_r.append(mse_period(R, r))
            mse_t.append(mean_squared_error(T, t))
            mae_r.append(mae_period(R, r))
            mae_t.append(np.absolute((T - t)).mean().item())
        except:
            # if this sample cannot produce reasonable result.
            degree_r.append(360.)
            mse_r.append(360.)
            mse_t.append(360.)
            mae_r.append(360.)
            mae_t.append(360.)

        # ADD 
        src_gt = R_ @ src + T
        src_eval  = r_ @ src + t
        add = torch.sum((src_gt - src_eval)**2, 0).sqrt().mean()

        # ADD_S
        val, _ = torch.sum((src_gt[:, None, :] - src_eval[:, :, None])**2, 0).sqrt().min(0)
        add_s = val.mean()

        th = .1 * diameter
        add_list.append(add < th)
        add_s_list.append(add_s < th)
    
    mAP_R5, mAP_R10, mAP_R15, mAP_R20, mAP_R25, mAP_R30 = mAP_th(degree_r)
    mAP_t5, mAP_t10, mAP_t15, mAP_t20, mAP_t25, mAP_t30 = mAP_th(mse_t, [0., 1.e-3, 5.e-3, 1.e-2, 5.e-2, 1.e-1, 5.e-1])
    metrics = {
        'mAP_R5': mAP_R5,
        'mAP_R10': mAP_R10,
        'mAP_R15': mAP_R15,
        'mAP_R20': mAP_R20,
        'mAP_R25': mAP_R25,
        'mAP_R30': mAP_R30,
        'mAP_t5': mAP_t5,
        'mAP_t10': mAP_t10,
        'mAP_t15': mAP_t15,
        'mAP_t20': mAP_t20,
        'mAP_t25': mAP_t25,
        'mAP_t30': mAP_t30,
        'mAP_R_mean': mAP_th(degree_r).mean(),
        'mAP_t_mean': mAP_th(mse_t, [0., 5., 10., 15.]).mean(),
        'add': np.asarray(add_list).sum()/len(add_list),
        'adds': np.asarray(add_s_list).sum()/len(add_s_list),
    }
    return metrics

def bop_vsd(source, target, rotation, translation, scores, info_dict, cpp_renderer):
    result = []
    for _, (score, src, tgt, R, T, depth, cam_K, obj_id, rgb) in enumerate(zip(scores.cpu(), 
                                                                source.cpu(), 
                                                                target.cpu(), 
                                                                rotation.cpu(), 
                                                                translation.cpu(), 
                                                                info_dict['depth'], 
                                                                info_dict['cam_K'],
                                                                info_dict['obj_id'],
                                                                info_dict['rgb'],
                                                                )):
        score[-1, -1] = np.inf
        val, row = torch.max(score, 0)
        col = torch.arange(score.shape[1])
        mask = (row != score.shape[0] - 1).type(torch.bool)
        src_m, tgt_m = src[:, row[mask]], tgt[:, col[mask]]
        r, t = calc_pose(src_m, tgt_m)
        R_, T_, r_, t_ = R, T, r, t

        tmp = Rotation.from_matrix(R.numpy())
        R = tmp.as_euler('ZYX', degrees=True)
        tmp = Rotation.from_matrix(r.numpy())
        r = tmp.as_euler('ZYX', degrees=True)

        # diameter {'xepose':148.4022, 'tudl':430.31, 'ycbv':172.063}
        res = pose_error.vsd(R_est=r_.numpy(), t_est=t.numpy(), R_gt=R_.numpy(), t_gt=T.numpy(), depth_test=depth.numpy(), K=cam_K.numpy(), \
                        delta=15, taus=list(np.arange(0.05, 0.51, 0.1)), \
                        normalized_by_diameter=True, diameter=172.063, \
                        renderer=cpp_renderer, obj_id=obj_id.numpy(), cost_type='step')

        # bop visual
        poses = [{'obj_id':obj_id.numpy(), 'R':r_.numpy(), 't':t_.numpy()}]
        visualization.vis_object_poses(poses=poses, K=cam_K.numpy(), renderer=cpp_renderer, rgb=rgb.numpy(), depth=depth.numpy(), \
            vis_rgb_path='./vis_rgb.jpg', \
            vis_depth_diff_path='./vis_depth_diff.jpg', \
            vis_rgb_resolve_visib=False)
        import ipdb; ipdb.set_trace()

def cls_metric(logits, labels):
    # accuracy
    pred = logits.max(dim=1)[1]
    acc_x = metrics.accuracy_score(labels[:,0].cpu().numpy(), pred[:,0].detach().cpu().numpy())
    acc_y = metrics.accuracy_score(labels[:,1].cpu().numpy(), pred[:,1].detach().cpu().numpy())
    acc_z = metrics.accuracy_score(labels[:,2].cpu().numpy(), pred[:,2].detach().cpu().numpy())

    acc = (acc_x + acc_y + acc_z) / 3.
    metrics_dict = {
        'acc': acc,
    }
    return metrics_dict