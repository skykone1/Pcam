from copy import deepcopy
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.spatial.transform.rotation import Rotation

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

import sys
sys.path.append('/home/hush/projects/bpnet-22-new')
from bpnet_utils.pytorch3d_utils import DepthRenderer, build_bijective_label, depth2xyz
from bpnet_utils.pytorch3d_utils import init_raster, verts2pt3d, generate_depth, voxelization
from bpnet_utils.mesh_utils import merge_and_normalize_mesh
from bpnet_utils.trimesh_utils import get_trimesh_zoo, trimesh_normalization, truncation, intersection, shearing, stretching

import lightconvpoint.nn as lcp_nn
from pcam.tool.transforms import ground_truth_attention


class GeoDesignDataset(Dataset):
    def __init__(self, args, partition, tuner_params=None):
        super().__init__()
        self.args = args
        self.partition = partition
        self.mesh_sampling_pts = 1024*4
        self.src_points = 1024
        self.tgt_points = 768

        if tuner_params is not None:
            self.scale = (.1, tuner_params['scale'])
            self.t = tuner_params['trans']
            self.R_aug = np.pi/tuner_params['rot']
            self.mesh_list = get_trimesh_zoo(tuner_params['num_obj'])
            self.shearing = (tuner_params['shear_x'], tuner_params['shear_y'], tuner_params['shear_z'])
            self.stretching = (tuner_params['stretch_x'], tuner_params['stretch_y'], tuner_params['stretch_z'])
            self.tuncation = tuner_params['truncation']
        else:
            # geometry design parameter
            self.scale = (.2, .8)
            self.t = .5
            self.R_aug = np.pi
            self.mesh_list = get_trimesh_zoo(3)
            self.shearing = (.2, .2, .2)
            self.stretching = (.2, .2, .2)
            self.truncation = 0.

        R_max = np.asarray([ np.pi,  np.pi/2,  np.pi])
        R_min = np.asarray([-np.pi, -np.pi/2, -np.pi])
        self.R_list = (R_max, R_min)
        self.t_max = np.asarray([ 30,  30, 1100]) / 100.
        self.t_min = np.asarray([-30, -30, 1000]) / 100.

    def __len__(self):
        # return 1000
        if self.partition != 'train':
            return 480
        else:
            return 4800
    
    def __getitem__(self, index: int):
        if self.partition != 'train':
            torch.manual_seed(index)
            torch.cuda.manual_seed_all(index)
            np.random.seed(index)
        # using rendering on the fly
        with torch.no_grad():
            _, rasterizer, self.C_inv = init_raster(img_size=320)
            depth_renderer = DepthRenderer(rasterizer)
        verts, faces, cam_R_m2c, cam_t_m2c, pts1, verts_check = self.prepare_mesh(index)

        mesh = Meshes(verts=[verts], faces=[faces]).cuda()
        depth = generate_depth(depth_renderer, mesh)
        pts2 = depth2xyz(depth[0, :, :, 0], self.C_inv)

        # save for the visualization.
        # import deepdish as dd
        # A = {
        #     'verts': verts_check.numpy(),
        #     'faces': faces.numpy().astype(np.int),
        #     'pts_tgt': pts2,
        # }
        # dd.io.save(f'./save_res/mesh/{index}.h5', A)

        # plot the depth map
        # import imageio
        # depth = depth[0, :, :, 0]
        # depth = ((depth - depth.min())/(depth.max() - depth.min())) * 255.
        # imageio.imwrite('depth.png', depth.astype(np.uint8))
        # import ipdb; ipdb.set_trace()

        # source, target = voxelization(pts1, pts2, self.args.th, self.src_points, self.tgt_points)
        source, target = voxelization(pts1, pts2, 5.e-2, self.src_points, self.tgt_points)
        
        R_gt, t_gt = cam_R_m2c.numpy(), cam_t_m2c.numpy()

        # from bpnet_utils.visual import plot_point_cloud
        # A = np.eye(4)
        # A[:3, :3] = R_gt
        # A[:3, 3] = t_gt.squeeze()
        # # plot_point_cloud(source, None, np.eye(4))
        # plot_point_cloud(source, target, A)
        # import ipdb; ipdb.set_trace()
        # import os;os._exit(1)

        # label = build_bijective_label(self.args, source, target, R_gt, t_gt) # out of date
        # label = build_bijective_label(source, target, R_gt, t_gt, self.args.th, self.args.val, self.args.bi_layer)
        info_dict = {
            'model_scale': 1.,
            'diameter': 1.,
            'index': index,
        }

        source, target = source.T, target.T
        search = lcp_nn.SearchQuantized(K=32, stride=1)
        indices1, source = search(torch.from_numpy(source).float().unsqueeze(0))
        indices2, target = search(torch.from_numpy(target).float().unsqueeze(0))
        source = source.squeeze().numpy()
        target = target.squeeze().numpy()

        ###########
        # Ground truth attention matrix for training
        ret_trans = np.identity(4)
        ret_trans[:3, :3] = R_gt
        ret_trans[:3, 3] = t_gt.squeeze()
        attention = ground_truth_attention(source.T, target.T, ret_trans)
        ###########

        euler_ab = Rotation.from_matrix(np.asarray(R_gt)).as_euler('zyx')
        euler_ba = -euler_ab[::-1]
        R_ba = R_gt.T
        translation_ba = -R_ba.dot(t_gt)

        t_gt, translation_ba = t_gt.squeeze(), translation_ba.squeeze()

        # # For Test
        # return source.astype('float32'), target.astype('float32'), R_gt.astype('float32'), \
        #     t_gt.astype('float32'), 0, info_dict
        
        # For Train
        return source.astype('float32'), target.astype('float32'), R_gt.astype('float32'), \
            t_gt.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
            euler_ab.astype('float32'), euler_ba.astype('float32'), \
            indices1.squeeze().numpy(), indices2.squeeze().numpy(), \
            attention.A, np.zeros((1, 3)), np.zeros((1, 3))



        # return source.T.astype('float32'), target.T.astype('float32'), R_gt.astype('float32'), t_gt.astype('float32'), label, info_dict

    def prepare_mesh(self, index):
        models_scale = 1.
        R_max, R_min = self.R_list
        euler_rand = np.random.uniform(R_min, R_max)
        R = Rotation.from_euler('zyx', euler_rand).as_matrix()
        t = np.random.uniform(self.t_min, self.t_max).reshape(3, 1)
        cam_R_m2c = torch.from_numpy(R).float()
        cam_t_m2c = torch.from_numpy(t).float()* models_scale
        R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_matrix()).float()

        meshes = deepcopy(self.mesh_list[index%len(self.mesh_list)])
        # meshes_trans = []
        # for mesh in meshes:
        #     verts, faces = mesh
        #     verts = verts

        #     scale = (torch.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]).float()
        #     translation = torch.from_numpy(np.random.uniform((-self.t, -self.t, -self.t), (self.t, self.t, self.t))).float()
        #     R_aug_part = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [self.R_aug, self.R_aug, self.R_aug])).as_matrix()).float()

        #     verts = scale * (R_aug_part @ (verts.T)).T + translation
        #     meshes_trans.append((verts, faces))
        # verts, faces = merge_and_normalize_mesh(meshes_trans)
        
        mesh_list = []
        for mesh in meshes:
            scale = np.random.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
            translation = np.random.uniform((-self.t, -self.t, -self.t), (self.t, self.t, self.t))
            R_aug_part = Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [self.R_aug, self.R_aug, self.R_aug])).as_matrix()
            mesh = truncation(mesh, self.truncation)
            # mesh = intersection(mesh)
            mesh.vertices = scale * (R_aug_part @ (mesh.vertices.T)).T + translation
            mesh = shearing(mesh, self.shearing)
            mesh = stretching(mesh, self.stretching)
            mesh_list.append((torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).float()))
            # mesh.export('test.ply')
            # import ipdb; ipdb.set_trace()
        # mesh = trimesh.boolean.boolean_automatic(mesh_list, operation='union')
        # mesh = trimesh_normalization(mesh)
        # verts, faces = torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).float()
        verts, faces = merge_and_normalize_mesh(mesh_list)

        verts = (R_aug @ (verts.T)).T
        verts_check = deepcopy(verts)

        # source pts
        ori_mesh = Meshes(verts=[verts], faces=[faces])
        pts1 = sample_points_from_meshes(ori_mesh, self.mesh_sampling_pts).squeeze(0)
        pts1 = pts1.numpy()

        verts = cam_R_m2c @ verts.T + cam_t_m2c
        verts = verts2pt3d(verts)
        verts = verts.T.float()
        return verts, faces, cam_R_m2c, cam_t_m2c, pts1, verts_check