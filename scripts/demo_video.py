"""Image demo script."""
import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
# Commented out to prevent pytorch3d import errors
# from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def get_video_info(in_file):
    stream = cv2.VideoCapture(in_file)
    assert stream.isOpened(), 'Cannot capture source'
    datalen = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(stream.get(cv2.CAP_PROP_FOURCC))
    fps = stream.get(cv2.CAP_PROP_FPS)
    frameSize = (int(stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoinfo = {'fourcc': fourcc, 'fps': fps, 'frameSize': frameSize}
    stream.release()

    return stream, videoinfo, datalen


def recognize_video_ext(ext=''):
    if ext == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
    elif ext == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    elif ext == 'mov':
        return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
    else:
        print("Unknown video format {}, will use .mp4 instead of it".format(ext))
        return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--video-name',
                    help='video name',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--save-pk', default=False, dest='save_pk',
                    help='save prediction', action='store_true')
parser.add_argument('--save-img', default=False, dest='save_img',
                    help='save prediction', action='store_true')

opt = parser.parse_args()

cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

res_keys = [
    'pred_uvd',
    'pred_xyz_17',
    'pred_xyz_29',
    'pred_xyz_24_struct',
    'pred_scores',
    'pred_camera',
    'pred_betas',
    'pred_thetas',
    'pred_phi',
    'pred_cam_root',
    'transl',
    'transl_camsys',
    'bbox',
    'height',
    'width',
    'img_path'
]
res_db = {k: [] for k in res_keys}

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)
hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

print('### Extract Image...')
video_basename = os.path.basename(opt.video_name).split('.')[0]

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)
if not os.path.exists(os.path.join(opt.out_dir, 'raw_images')):
    os.makedirs(os.path.join(opt.out_dir, 'raw_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_images'))
if not os.path.exists(os.path.join(opt.out_dir, 'res_2d_images')) and opt.save_img:
    os.makedirs(os.path.join(opt.out_dir, 'res_2d_images'))

_, info, _ = get_video_info(opt.video_name)
video_basename = os.path.basename(opt.video_name).split('.')[0]

savepath = f'./{opt.out_dir}/res_{video_basename}.mp4'
savepath2d = f'./{opt.out_dir}/res_2d_{video_basename}.mp4'
info['savepath'] = savepath
info['savepath2d'] = savepath2d

write_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
write2d_stream = cv2.VideoWriter(
    *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])
if not write_stream.isOpened():
    print("Try to use other video encoders...")
    ext = info['savepath'].split('.')[-1]
    fourcc, _ext = recognize_video_ext(ext)
    info['fourcc'] = fourcc
    info['savepath'] = info['savepath'][:-4] + _ext
    info['savepath2d'] = info['savepath2d'][:-4] + _ext
    write_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
    write2d_stream = cv2.VideoWriter(
        *[info[k] for k in ['savepath2d', 'fourcc', 'fps', 'frameSize']])

assert write_stream.isOpened(), 'Cannot open video for writing'
assert write2d_stream.isOpened(), 'Cannot open video for writing'

os.system(f'ffmpeg -y -i {opt.video_name} {opt.out_dir}/raw_images/{video_basename}-%06d.png')

files = os.listdir(f'{opt.out_dir}/raw_images')
files.sort()

img_path_list = []

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        img_path = os.path.join(opt.out_dir, 'raw_images', file)
        img_path_list.append(img_path)

prev_box = None

# Renderer removed to bypass PyTorch3D requirement
# renderer = None
# smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

print('### Run Model...')
idx = 0
for img_path in tqdm(img_path_list):
    dirname = os.path.dirname(img_path)
    basename = os.path.basename(img_path)

    with torch.no_grad():
        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        if prev_box is None:
            tight_bbox = get_one_box(det_output)  # xyxy
            if tight_bbox is None:
                continue
        else:
            tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

        prev_box = tight_bbox

        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )
        uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        transl = pose_output.transl.detach()

        # ==========================================
        # 3D MESH RENDERER BYPASS
        # ==========================================
        # We skip the entire 3D mesh rendering block.
        # To keep the video writer functional without crashing, 
        # we simply pass the raw frame as the 3D output.
        image_vis = cv2.cvtColor(input_image.copy(), cv2.COLOR_RGB2BGR)

        if opt.save_img:
            idx += 1
            res_path = os.path.join(opt.out_dir, 'res_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, image_vis)
        write_stream.write(image_vis)

# ==========================================
        # 2D SKELETON VISUALIZATION
        # ==========================================
        bbox_xywh = xyxy2xywh(bbox)
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]
        
        pts_np = pts.cpu().numpy()
        
        image = input_image.copy()

        # Define skeleton links for HybrIK's SMPL 29-point format
        # Joints 0-23 are standard SMPL. Joints 24-28 are facial points.
        skeleton_links = [
            # Spine and Head (Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck -> Head)
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
            
            # Left Leg (Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot)
            (0, 1), (1, 4), (4, 7), (7, 10),(10, 27),
            
            # Right Leg (Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot)
            (0, 2), (2, 5), (5, 8), (8, 11),(11, 28),
            
            # Left Arm (Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist -> L_Hand)
            (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),(22, 25),
            
            # Right Arm (Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist -> R_Hand)
            (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),(23, 26), 
            
            # Facial points: Nose(24), R_Eye(25), L_Eye(26), R_Ear(27), L_Ear(28)
            (15, 24)
        ]

        # Draw the bones (lines) manually using the numpy array
        for link in skeleton_links:
            p1 = tuple(pts_np[link[0]].astype(int))
            p2 = tuple(pts_np[link[1]].astype(int))
            # Draw line with color (cyan) and thickness 2
            cv2.line(image, p1, p2, (255, 255, 0), 2)
        # ==========================================
        # MODO DEBUG: Escribir el número de cada punto
        # ==========================================
        for i, pt in enumerate(pts_np):
            x, y = int(pt[0]), int(pt[1])
            # Dibujar un pequeño círculo rojo en el punto
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            # Escribir el índice (número) del punto en color blanco
            cv2.putText(image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw bounding box and points over the lines
        bbox_img = vis_2d(image, tight_bbox, pts_np)
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        write2d_stream.write(bbox_img)

        if opt.save_img:
            res_path = os.path.join(
                opt.out_dir, 'res_2d_images', f'image-{idx:06d}.jpg')
            cv2.imwrite(res_path, bbox_img)

        # ==========================================
        # SAVE JOINT DATA TO PICKLE
        # ==========================================
        if opt.save_pk:
            assert pose_input.shape[0] == 1, 'Only support single batch inference for now'

            pred_xyz_jts_17 = pose_output.pred_xyz_jts_17.reshape(
                17, 3).cpu().data.numpy()
            pred_uvd_jts = pose_output.pred_uvd_jts.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_29 = pose_output.pred_xyz_jts_29.reshape(
                -1, 3).cpu().data.numpy()
            pred_xyz_jts_24_struct = pose_output.pred_xyz_jts_24_struct.reshape(
                24, 3).cpu().data.numpy()
            pred_scores = pose_output.maxvals.cpu(
            ).data[:, :29].reshape(29).numpy()
            pred_camera = pose_output.pred_camera.squeeze(
                dim=0).cpu().data.numpy()
            pred_betas = pose_output.pred_shape.squeeze(
                dim=0).cpu().data.numpy()
            pred_theta = pose_output.pred_theta_mats.squeeze(
                dim=0).cpu().data.numpy()
            pred_phi = pose_output.pred_phi.squeeze(dim=0).cpu().data.numpy()
            pred_cam_root = pose_output.cam_root.squeeze(dim=0).cpu().numpy()
            img_size = np.array((input_image.shape[0], input_image.shape[1]))

            transl_camsys = transl.clone()
            transl_camsys = transl_camsys * 256 / bbox_xywh[2]

            res_db['pred_xyz_17'].append(pred_xyz_jts_17)
            res_db['pred_uvd'].append(pred_uvd_jts)
            res_db['pred_xyz_29'].append(pred_xyz_jts_29)
            res_db['pred_xyz_24_struct'].append(pred_xyz_jts_24_struct)
            res_db['pred_scores'].append(pred_scores)
            res_db['pred_camera'].append(pred_camera)
            res_db['pred_betas'].append(pred_betas)
            res_db['pred_thetas'].append(pred_theta)
            res_db['pred_phi'].append(pred_phi)
            res_db['pred_cam_root'].append(pred_cam_root)
            res_db['transl'].append(transl[0].cpu().data.numpy())
            res_db['transl_camsys'].append(transl_camsys[0].cpu().data.numpy())
            res_db['bbox'].append(np.array(bbox))
            res_db['height'].append(img_size[0])
            res_db['width'].append(img_size[1])
            res_db['img_path'].append(img_path)

if opt.save_pk:
    n_frames = len(res_db['img_path'])
    for k in res_db.keys():
        print(k)
        res_db[k] = np.stack(res_db[k])
        assert res_db[k].shape[0] == n_frames

    with open(os.path.join(opt.out_dir, 'res.pk'), 'wb') as fid:
        pk.dump(res_db, fid)

write_stream.release()
write2d_stream.release()