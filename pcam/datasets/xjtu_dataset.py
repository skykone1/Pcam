from copy import deepcopy
import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform.rotation import Rotation
import os, sys
sys.path.append('/home/dz/2021_code_nas/XEPose/bop_toolkit')

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout

from pytorch3d.structures import Meshes

from pytorch3d.ops import sample_points_from_meshes

sys.path.append('/home/hush/projects/bpnet-22')
from bpnet_utils.pytorch3d_utils import DepthRenderer, build_bijective_label, depth2xyz, load_modelnet40_asymmetry_manually
from bpnet_utils.pytorch3d_utils import init_sampling_raster, Euler_list, sampling_from_mesh_surface
from bpnet_utils.pytorch3d_utils import init_raster, verts2pt3d, generate_depth, voxelization
from bpnet_utils.bop_utils import get_bop_dataset, get_bop_models

import lightconvpoint.nn as lcp_nn
from pcam.tool.transforms import ground_truth_attention

class XJTUBase(Dataset):
    def __init__(self, dataset, partition, split_type, format=None, pose_type=None, subset=None):
        super().__init__()
        self.dataset = dataset
        self.partition = partition
        self.format = format
        self.pose_type = pose_type
        self.subset = subset
        self.sampling_type = 'from_mesh'
        # self.sampling_type = 'from_surface_on_the_fly'
        print(self.sampling_type)
        self.rot_aug = True # TODO: for debug, control the real scene dataset rotation augmentation.

        if format == 'pytorch3d':
            self.data_all = load_modelnet40_asymmetry_manually(self.partition, category=None)
            # self.data = load_modelnet40_auto_aligned(self.partition, category=['guitar.h5'])
            asym_idx, plane_sym_idx, axis_sym_idx, asym_clear_idx = [], [], [], []
            for idx, item in enumerate(self.data_all):
                if item[3] == '0':
                    asym_idx.append(idx)
                elif item[3] == '1':
                    plane_sym_idx.append(idx)
                elif item[3] == '2':
                    axis_sym_idx.append(idx)
                elif item[3] == '4':
                    asym_clear_idx.append(idx)
                else:
                    raise 'symmetry type error'

            # print(f'asym_obj:{len(asym_idx)} '
            #     f'plane_sym_obj:{len(plane_sym_idx)} '
            #     f'axis_sym_obj:{len(axis_sym_idx)} '
            #     f'asym_clear_obj:{len(asym_clear_idx)}')

            if self.subset == 'all':
                subset_idx = asym_idx + plane_sym_idx + axis_sym_idx + asym_clear_idx
            elif self.subset == 'asym_3055':
                subset_idx = asym_idx + asym_clear_idx
            elif self.subset == 'plane_sym':
                subset_idx = plane_sym_idx
            elif self.subset == 'axis_sym':
                subset_idx = axis_sym_idx
            elif self.subset == 'sym_all':
                subset_idx = axis_sym_idx + plane_sym_idx
            elif self.subset == 'asym_clear':
                subset_idx = asym_clear_idx
            elif self.subset == 'asym_500':
                subset_idx = np.random.choice(asym_idx+asym_clear_idx, size=500, replace=False).tolist()
            elif self.subset == 'sym_1022':
                subset_idx = np.random.choice(axis_sym_idx + plane_sym_idx, size=1022, replace=False).tolist()
            else:
                raise 'subset type error'
            # for idx, item in enumerate(self.data):
            #     if item[-1] == 'car_0265_1.off':
            #         print(idx)
            # import ipdb; ipdb.set_trace()
            
            # shrink the data list according to subset_idx
            self.data = [item for idx, item in enumerate(self.data_all) if idx in subset_idx]

        elif format == 'bopdataset' and pose_type == 'read_file':
            self.data, self.dp_split, self.mp_split = get_bop_dataset(dataset, partition, split_type)
        elif format == 'bopdataset' and pose_type == 'random':
            self.data = get_bop_models(dataset_list=['lmo', 'tudl', 'ycbv'])
        else:
            raise('not implement yet.')
        print(dataset, partition, split_type, len(self.data))

        R_max = np.asarray([ np.pi,  np.pi/2,  np.pi])
        R_min = np.asarray([-np.pi, -np.pi/2, -np.pi])
        self.R_list = (R_max, R_min)

        self.t_max = np.asarray([ 30,  30, 1100])
        self.t_min = np.asarray([-30, -30, 1000])

        if self.sampling_type == 'from_surface_on_the_fly':
            # init the mesh surface sampling render.
            self.euler_list = Euler_list()
            rasterizer, self.simpling_C_inv = init_sampling_raster()
            self.sampling_render = DepthRenderer(rasterizer)
    
    @staticmethod
    def crop_mask(mask):
        mask_crop = cv2.erode(mask, kernel=np.ones((2, 2), np.uint8), iterations=2)
        return mask_crop

    def prepare_mesh(self, index):
        ''' 
        using index load the model then convert to mesh.'''

        if self.format == 'pytorch3d':
            verts_ori, faces_ori, _, _, _, obj_name = self.data[index]
            # offline surface sampling
            # verts_ori, faces_ori, _, _, _, obj_name, surface_pts = self.data[index]
            verts, faces = deepcopy(verts_ori), deepcopy(faces_ori)
            if self.partition == 'train':
                R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_dcm().astype('float32'))
                verts = (R_aug @ (verts.T)).T
            # verts *= 100 # scale the data to make the bopdataset pipeline works smoothly.
        elif self.format == 'bopdataset' and self.pose_type == 'read_file':
            scene_gt = self.data[index]['scene_gt']
            triangle_mash = o3d.io.read_triangle_mesh((self.mp_split['model_tpath'].format(obj_id=scene_gt['obj_id'])))
            verts = torch.from_numpy(np.asarray(triangle_mash.vertices))
            faces = torch.from_numpy(np.asarray(triangle_mash.triangles))
            
            # depth = inout.load_depth(self.dp_split['depth_tpath'].format(scene_id=self.data[index]['scene_id'], im_id=self.data[index]['im_id']))
            # depth = ((depth - depth.min())/(depth.max() - depth.min())) * 255.
            # import imageio
            # imageio.imwrite('depth_real.png', depth.astype(np.uint8))
        elif self.format == 'bopdataset' and self.pose_type == 'random':
            '''
            Here we use the bopdataset's model for training'''
            index %= len(self.data) # for the number of object in bop_challenge is always very small.
            mesh = deepcopy(self.data[index]['mesh'])
            verts = torch.from_numpy(np.asarray(mesh.vertices))
            # faces = torch.from_numpy(np.asarray(mesh.triangles)) # for o3d
            faces = torch.from_numpy(np.asarray(mesh.faces)) 
        else:
            raise('not implement yet.')
        verts = verts.float()
        faces = faces.int()

        # self.to_meter = 0.001 # mm to m
        # careful about the unused facet vertice.
        if self.dataset == 'modelnet40' :
            models_scale = 1 / 100 
        else:
            models_scale = 1 / (max(verts.max(), np.abs(verts.min())))
            verts *= models_scale

        if self.sampling_type == 'from_mesh':
            # 1. uniformly sampling from the surface of the mesh model.
            ori_mesh = Meshes(verts=[verts], faces=[faces])
            pts1 = sample_points_from_meshes(ori_mesh, self.mesh_sampling_pts).squeeze(0)
            pts1 = pts1.numpy()
        elif self.sampling_type == 'from_surface_offline':
            # 2. randomly sampling from the visible surface of the mesh model. (offline)
            pts_idx = np.arange(surface_pts.shape[0])
            pts1 = surface_pts[np.random.choice(pts_idx, self.mesh_sampling_pts, replace=False)]
            if self.partition == 'train': pts1 = (R_aug @ (pts1.T)).T
        elif self.sampling_type == 'from_surface_on_the_fly':
            # 3. randomly sampling from the visible surface of the mesh model. (on-the-fly)
            surface_pts = sampling_from_mesh_surface((verts, faces), self.sampling_render, self.simpling_C_inv, self.euler_list.euler_12)
            pts_idx = np.arange(surface_pts.shape[0])
            pts1 = surface_pts[np.random.choice(pts_idx, min(self.mesh_sampling_pts, len(pts_idx)), replace=False)]
        else:
            raise 'don not have this sampling method.'

        if self.pose_type == 'read_file':
            cam_R_m2c = torch.from_numpy(scene_gt['cam_R_m2c']).float() # pose of the object.
            cam_t_m2c = torch.from_numpy(scene_gt['cam_t_m2c']).float() * models_scale
            if self.partition == 'train' and self.rot_aug is True:
                R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_dcm().astype('float32'))
                pts1 = (R_aug @ (verts.T)).T
                pts1 = pts1.numpy()
                cam_R_m2c = cam_R_m2c @ R_aug.T
        elif self.pose_type == 'random':
            # R_max, R_min = self.R_list[np.random.randint(len(self.R_list))]
            R_max, R_min = self.R_list
            euler_rand = np.random.uniform(R_min, R_max)
            R = Rotation.from_euler('zyx', euler_rand).as_dcm()
            t = np.random.uniform(self.t_min, self.t_max).reshape(3, 1)
            cam_R_m2c = torch.from_numpy(R).float()
            cam_t_m2c = torch.from_numpy(t).float()* models_scale
        else:
            raise('not implement yet.')

        # transfrom the verts for rendering the target depth map.
        verts = cam_R_m2c @ verts.T + cam_t_m2c
        # verts to pytorch3d coordinate
        verts = verts2pt3d(verts)
        # Create a Meshes object. Here we have only one mesh in the batch.
        verts = verts.T.float()
        diameter = self.data[index]['diameter'] if self.format == 'bopdataset' else 1.
        model_info = {
            'scale': models_scale,
            'diameter': diameter,
        }
        return verts, faces, cam_R_m2c, cam_t_m2c, pts1, model_info


class XJTU_Dataset(XJTUBase):
    '''
    Rendering the training sample on-the-fly. Using the xjtu camera parameters.
    self.mask_type = ['render', 'gt', 'yolact']
    '''
    def __init__(self, args, partition=None, split_type=None, dataset='modelnet40', format='pytorch3d', pose_type='random', mask_type='render', subset=None, load_cache=False):
        self.args = args
        super().__init__(dataset, partition, split_type, format, pose_type, subset)

        self.mesh_sampling_pts = 1024*4
        self.src_points = 1024
        self.tgt_points = 768
        self.dataset = dataset
        self.load_cache = load_cache
        self.mask_type = mask_type

        # print(self.dataset, self.partition, self.split_type, len(self.data))
        
        # _, rasterizer, C_inv = init_raster(img_size=320)
        # self.depth_renderer = DepthRenderer(rasterizer)
        # self.C_inv = C_inv

        if self.partition == 'test' and self.dataset == 'xepose' and self.load_cache == True:
            print('Using xepose cache!')
            self.xepose_cache = np.load('/cache/xepose_test_203_204_cache.npy', allow_pickle=True)
            print('Xepose cache load complete.')
        
    def __len__(self):
        if len(self.data) < 500 and self.partition == 'train':
            # return 32*500
            return 9600
        
        return len(self.data)


    def __getitem__(self, index: int):
        if self.load_cache == True:
            return self.xepose_cache[index]['src'], self.xepose_cache[index]['tgt'], self.xepose_cache[index]['R_gt'], \
                self.xepose_cache[index]['t_gt'], self.xepose_cache[index]['label'], self.xepose_cache[index]['info_dict']

        if self.partition != 'train':
            torch.manual_seed(index)
            torch.cuda.manual_seed_all(index)
            np.random.seed(index)
        if self.mask_type == 'render':
            # using rendering on the fly
            with torch.no_grad():
                _, rasterizer, self.C_inv = init_raster(img_size=320)
                depth_renderer = DepthRenderer(rasterizer)
            verts, faces, cam_R_m2c, cam_t_m2c, pts1, model_info = self.prepare_mesh(index)
            models_scale = model_info['scale']
            mesh = Meshes(verts=[verts], faces=[faces]).cuda()
            depth = generate_depth(depth_renderer, mesh)
            pts2 = depth2xyz(depth[0, :, :, 0], self.C_inv)

            # import imageio
            # depth = depth[0, :, :, 0]
            # depth = ((depth - depth.min())/(depth.max() - depth.min())) * 255.
            # imageio.imwrite('depth_real.png', depth.astype(np.uint8))
            # import ipdb; ipdb.set_trace()
            info_dict = {}
        elif self.mask_type == 'gt':
            _, _, cam_R_m2c, cam_t_m2c, pts1, model_info = self.prepare_mesh(index)
            models_scale = model_info['scale']
            scene_id = self.data[index]['scene_id']
            im_id = self.data[index]['im_id']
            gt_id = self.data[index]['gt_id']
            scene_camera = self.data[index]['scene_camera']
            mask = inout.load_im(self.dp_split['mask_visib_tpath'].format(scene_id=scene_id, im_id=im_id, gt_id=gt_id)) 
            depth = inout.load_depth(self.dp_split['depth_tpath'].format(scene_id=scene_id, im_id=im_id))
            depth *= scene_camera['depth_scale']  # Convert to [mm].
            mask = np.array(mask) / 255
            if self.dataset not in ['modelnet40', 'xepose']: mask = self.crop_mask(mask)
            depth = np.array(depth) * mask
            
            # refine process for real dataset.
            if self.dataset not in ['modelnet40', 'xepose']:
                tgt = self.refine_mask_median(mask, depth, scene_camera, th=np.asscalar(1./models_scale))
                # tgt = self.refine_mask_median(mask, depth, scene_camera, th=200.)
                # tgt = self.refine_mask_convert_to_cam(mask, depth, scene_camera, th=2.)
            else:
                tgt = self.convert_to_cam(depth, scene_camera['cam_K'])

            pts2 = tgt * models_scale.numpy()
            if self.partition == 'train' and self.rot_aug is True:
                R_aug = torch.from_numpy(Rotation.from_euler('zyx', np.random.uniform([0., 0., 0.], [np.pi/2, np.pi/2, np.pi/2])).as_dcm().astype('float32'))
                pts2 = (R_aug @ (pts2.T)).T.numpy()
                cam_R_m2c = R_aug @ cam_R_m2c
                cam_t_m2c = R_aug @ cam_t_m2c
            
            info_dict = {
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': self.data[index]['scene_gt']['obj_id'],
            }
        else:
            raise('mask_type not implement yet')

        # voxelize the point cloud.
        source, target = voxelization(pts1, pts2, 5e-2, self.src_points, self.tgt_points)
        R_gt, t_gt = cam_R_m2c.numpy(), cam_t_m2c.numpy()

        # if info_dict['scene_id'] == 1 and info_dict['im_id'] == 3075 and info_dict['obj_id']==1:
        # from bpnet_utils.visual import plot_point_cloud
        # A = np.eye(4)
        # A[:3, :3] = R_gt
        # A[:3, 3] = t_gt.squeeze()
        # # plot_point_cloud(source, None, np.eye(4))
        # plot_point_cloud(source, target, A)
        # import ipdb; ipdb.set_trace()
        # import os;os._exit(1)

        # A = {}
        # A['th'] = self.args.th
        # A['bi_layer'] = 'ot'
        # A['val'] = self.args.val

        # label = build_bijective_label(self.args, source, target, R_gt, t_gt)
        # # label = build_bijective_label_cuda(self.args, source, target, R_gt, t_gt)
        info_dict['model_scale'] = models_scale
        info_dict['diameter'] = model_info['diameter']
        # return source.T.astype('float32'), target.T.astype('float32'), R_gt.astype('float32'), t_gt.astype('float32'), label, info_dict
        
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

        euler_ab = Rotation.from_dcm(np.asarray(R_gt)).as_euler('zyx')
        euler_ba = -euler_ab[::-1]
        R_ba = R_gt.T
        translation_ba = -R_ba.dot(t_gt)

        t_gt, translation_ba = t_gt.squeeze(), translation_ba.squeeze()

        # For Test
        return source.astype('float32'), target.astype('float32'), R_gt.astype('float32'), \
            t_gt.astype('float32'), 0, info_dict
        
        # For Train
        return source.astype('float32'), target.astype('float32'), R_gt.astype('float32'), \
            t_gt.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
            euler_ab.astype('float32'), euler_ba.astype('float32'), \
            indices1.squeeze().numpy(), indices2.squeeze().numpy(), \
            attention.A, np.zeros((1, 3)), np.zeros((1, 3))


    def convert_to_cam(self, depth, cam_K):
        x, y = np.where(depth > 0)
        z = depth[x, y]
        C_inv = np.linalg.inv(cam_K)
        ones = np.ones_like(x)
        xy1 = np.stack([y, x, ones], axis=0)
        xy1_cam = C_inv @ xy1
        tgt = (xy1_cam * z[None]).T
        return tgt

    def refine_mask_convert_to_cam(self, mask, depth, scene_camera, th=1.2):
        '''
        Get the center of the target in the depth map. When converting it to the world
        coordinate system, using the range [-1, 1] to remove the 3D point which out of
        the range.
        '''
        x, y = np.where(mask > 0)
        xy = np.stack([x, y], axis=1)
        mask_coor = np.zeros_like(x).astype(np.bool)
        mask_coor[np.square(xy - xy.mean(0)).sum(1).argmin()] = True
        z = depth[x, y]
        C_inv = np.linalg.inv(scene_camera['cam_K'])
        ones = np.ones_like(x)
        xy1 = np.stack([y, x, ones], axis=0)
        xy1_cam = C_inv @ xy1
        tgt = (xy1_cam * z[None]).T
        tgt_center = tgt[mask_coor] # 2D geometric centeroid
        mask_z_pos = (tgt[:, 2] - tgt_center[:, 2]) < th
        mask_z_neg = (tgt[:, 2] - tgt_center[:, 2]) > - th
        return tgt[np.logical_and(mask_z_pos, mask_z_neg)]

    def refine_mask_median(self, mask, depth, scene_camera, th=200):
        '''
        Get the median depth value in the object's depth map. Then using it to
        get rid of the outlier which far from the target object's surface.
        '''

        x, y = np.where(mask > 0)
        z = depth[x, y]
        C_inv = np.linalg.inv(scene_camera['cam_K'])
        ones = np.ones_like(x)
        xy1 = np.stack([y, x, ones], axis=0)
        xy1_cam = C_inv @ xy1
        tgt = (xy1_cam * z[None]).T

        tgt_median = np.median(tgt[:, 2])
        mask = np.abs(tgt[:, 2] - tgt_median) < th 
        return tgt[mask]

def save_cache():
    class Args():
        th = 5e-2
        val = 1
        bi_layer = 'ot'

    args = Args()
    dataset = XJTU_Dataset(args=args, partition='test', dataset='xepose', mask_type='gt', format='bopdataset', pose_type='read_file')
    total = []
    print(dataset.__len__())
    # for i in range(0, dataset.__len__()):
    for i in range(0, 10):
        src, tgt, R_gt, t_gt, label, info_dict = dataset.__getitem__(i)
        total.append({
            'src': src,
            'tgt': tgt,
            'R_gt': R_gt,
            't_gt': t_gt,
            # 'label': label,
            'info_dict': info_dict
        })
        if i % 100 == 0:
            print(i)

    ndarray = np.array(total)
    np.save('./cache/file_name.npy', ndarray, allow_pickle=True)
    

def test_new_dataloader():
    class Args():
        th = 5e-2
        val = 1
        bi_layer = 'ot'

    args = Args()
    dataset = XJTU_Dataset(args=args, partition='train', dataset='modelnet40', subset='all', format='pytorch3d', pose_type='random', mask_type='render')
    for i in range(0, 10):
        xx = dataset.__getitem__(i)
        print('1')

if __name__ == '__main__':
    # save_cache()
    test_new_dataloader()