import os
import glob
from itertools import product
import sys
sys.path.append(os.path.abspath("../.."))

import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import softmax, expit
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from scipy.ndimage.morphology import grey_erosion

import tensorflow as tf
import keras.backend as K
from keras.utils import Sequence, to_categorical
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPool2D, Input, Lambda, UpSampling2D
from keras.layers import Concatenate
from keras.optimizers import RMSprop, Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback

from pycocotools import coco
import pycocotools.mask as mask_util
import imgaug as ia
import imgaug.augmenters as iaa

from src.utils import bbox_utils as bbu
from src.utils import image_utils as imu


def GetDensePoseMask(Polys):
    """Decodes densepose annotations into 256x256 mask of 15 integers (background and 14 parts).
    
    # Returns: np.array with shape (256, 256) and float32 dtype."""
    MaskGen = np.zeros([256, 256], dtype=np.float32)
    for i in range(1, 15):
        if Polys[i-1]:
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask > 0] = i
    return MaskGen


def keypoints2numpy(keypoints):
    """Converts a list of COCO keypoints into numpy array.
    Each row of resulting matrix is a point, that has vertical and horizontal coordinates, plus a visibility flag (0, 1, 2).
    
    # Returns: np.array with shape (N, 3) and float32 dtype."""
    arr = np.array(keypoints, dtype=np.float32).reshape((-1, 3))
    return arr[:, [1, 0, 2]]  # first axis is now vertical

assert np.all(
    keypoints2numpy([0, 0, 1, 10, 10, 1, 20, 20, 2, 30 , 30, 0, 45, 10, 1]) == 
    np.array([[0, 0, 1], [10, 10, 1], [20, 20, 2], [30, 30, 0], [10, 45, 1]], dtype=np.float32)
)


def get_yxhw(kpts_numpy):
    """Get a `yxhw` bounding box around an array of COCO keypoints.
    
    # Returns: a tuple (y, x, h, w)."""
    kpts_vis = kpts_numpy[kpts_numpy[:, 2] > 0, :]
    y1, x1, y2, x2 = kpts_vis[:, 0].min(), kpts_vis[:, 1].min(), kpts_vis[:, 0].max(), kpts_vis[:, 1].max()
    return (y1, x1, y2-y1, x2-x1)

assert get_yxhw(np.array([[0, 0, 1], [10, 10, 1], [20, 20, 2], [30, 30, 0], [10, 45, 1]], dtype=np.float32)) == (0., 0., 20., 45.)


def points_to_heatmap(points, original_size, desired_size, sigma):
    """Puts pixels on an image and then blur them to produce a heatmap.
    `points` is expected to have (n, 4) shape: vertical, horizontal coordinates, COCO visibility, joint id.
    
    # Returns: np.array of shape `(*desired_size, 17)` float32 dtype.
    """
    num_joints_coco = 17
    heatmap = np.zeros((*desired_size, num_joints_coco), dtype=np.float32)
    for i in range(points.shape[0]):
        y, x, j = points[i, 0], points[i, 1], points[i, 3]
        y = y / original_size[0] * desired_size[0]
        x = x / original_size[1] * desired_size[1]
        y, x = int(round(y)), int(round(x))
        if x >=0 and y >= 0 and y < desired_size[0] and x < desired_size[1]:
            heatmap[y, x, int(j)] += 1.
        
    for j in range(num_joints_coco):
        heatmap[:, :, j] = gaussian_filter(heatmap[:, :, j], sigma=sigma, order=0, output=None, mode='constant', cval=0.0, truncate=sigma+1.5)
    return heatmap


def calculate_heatmap_peak(sigma):
    size = 1
    heatmap = np.zeros((size, size), dtype=np.float32)
    heatmap[size//2, size//2] = 1.
    return gaussian_filter(heatmap, sigma=sigma, order=0, output=None, mode='constant', cval=0.0, truncate=sigma+1.5)[size//2, size//2]

assert np.isclose(calculate_heatmap_peak(1.), 0.1592411, rtol=0., atol=1e-6)


def points_to_heatmap_bce_approach(points, original_size, desired_size, sigma):
    """Puts pixels on an image and then blur them to produce a heatmap.
    `points` is expected to have (n, 4) shape: vertical, horizontal coordinates, COCO visibility, joint id.
    
    # Returns: np.array of shape `(*desired_size, 17)` float32 dtype.
    """
    max_val = calculate_heatmap_peak(sigma)
    heatmap = points_to_heatmap(points, original_size, desired_size, sigma)
    heatmap /= max_val  # rescale heatmap so that joint center has value one
    heatmap[heatmap > 1.] = 1.  # clip values above 1.
    return heatmap


def img_id_from_person_id(coco_obj, person_id):
    annotation = coco_obj.anns[person_id]
    return annotation['image_id']

# assert img_id_from_person_id(DP_COCO, 186574) == 458755

def get_image(coco_obj, img_id, base_path, grayscale=True, size=None):
    """Given a COCO dataset object, retrieve an image by `img_id`.
    
    # Returns: np.uint8 image of shape `(H, W, 3)` or `(H, W)` (depends on `greyscale` parameter)."""
    img_fn = coco_obj.imgs[img_id]['file_name']
    image = cv2.imread(os.path.join(base_path, img_fn), cv2.IMREAD_COLOR)
    if size is not None:
        image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image[:, :, ::-1]  # bgr2rgb
    return image

num_joints_coco = 17

def get_all_keypoints(coco_obj, img_id, size=None):
    """Given a COCO dataset object, retrieve the coordinates of skeleton joints
    that belong to all people in the image with `image_id`.
    
    # Returns: np.array of shape (N, 4) and float32 dtype, where each row is a skeleton joint.
    And 4 numbers the row consists of are: vertical and horizontal coordinates, COCO visibility, joint index."""
    all_anns = coco_obj.imgToAnns[img_id]
    all_keypoints = []
    
    for pann in all_anns:
        if pann['keypoints']:
            pts = keypoints2numpy(pann['keypoints'])
            all_keypoints.append(pts)
    
    # transform visible keypoints into (-1, 2) matrix plus an array with joint index
    joint_ids = np.arange(num_joints_coco).reshape((-1, 1))  # a column of joint indices
    joint_ids = np.tile(joint_ids, (len(all_keypoints), 1))  # tiled several times to match gt
    all_keypoints = np.concatenate(all_keypoints, axis=0)  # concatenate all gt arrays
    # assert all_keypoints.shape[0] % num_joints_coco == 0
    all_keypoints = np.concatenate([all_keypoints, joint_ids], axis=1)  # attach joint ids
    all_keypoints = all_keypoints[all_keypoints[:, 2] > 0]  # filter absent joints
    
    if size is not None:  # TODO
        pass
    
    return all_keypoints


def get_keypoint_loss_region(coco_obj, img_id, size=None):
    """Given a COCO dataset object, retrieve binary mask of the region that we propagate skeleton joints loss through.
    Loss is not propagated trough the pixels of humans with no keypoints labeled.
    
    # Returns: np.array of shape (H, W) and float32 dtype."""
    img_meta = coco_obj.imgs[img_id]
    all_anns = coco_obj.imgToAnns[img_id]    
    loss_region = np.ones((img_meta['height'],  img_meta['width']), dtype=np.bool)
    for pann in all_anns:
        if pann['num_keypoints'] == 0:
            rle = mask_util.frPyObjects(pann['segmentation'], img_meta['height'],  img_meta['width'])
            m = mask_util.decode(rle).astype(np.bool)
            m = np.logical_or.reduce(m, axis=2) if len(m.shape) == 3 else m
            loss_region &= ~m
    
    if size is not None:  # TODO?
        pass
    
    return loss_region.astype(np.float32)

def get_binary_seg(coco_obj, img_id, size=None):
    """Given a COCO dataset object, retrieve binary segmentation of the humans
    in the image with `img_id`.
    
    # Returns: np.array of shape (H, W) and float32 dtype."""
    img_meta = coco_obj.imgs[img_id]
    all_anns = coco_obj.imgToAnns[img_id]
    seg_mask = np.zeros((img_meta['height'],  img_meta['width']), dtype=np.bool)
    
    for pann in all_anns:
        rle = mask_util.frPyObjects(pann['segmentation'], img_meta['height'],  img_meta['width'])
        m = mask_util.decode(rle).astype(np.bool)
        seg_mask |= np.logical_or.reduce(m, axis=2) if len(m.shape) == 3 else m
    
    if size is not None:  # TODO
        pass
    
    return seg_mask.astype(np.float32)

num_seg_patches = 15

def get_densepose_part_segments(coco_obj, img_id, size=None):
    """Given a COCO dataset object, retrieve Densepose body part annotations as a categorical mask.
    
    # Returns: np.array of shape (H, W, 15) and float32 dtype."""
    img_meta = coco_obj.imgs[img_id]
    # load anns for that image
    anns = coco_obj.imgToAnns[img_id]
    
    # M = []
    M = np.zeros((img_meta['height'], img_meta['width'], num_seg_patches), dtype=np.float32)
    for ann in anns:
        bbr =  np.array(ann['bbox'], dtype=np.int)
        if 'dp_masks' in ann.keys():
            Mask = GetDensePoseMask(ann['dp_masks'])

            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0]+bbr[2], bbr[1]+bbr[3]
            x2 = min(x2, img_meta['width'])
            y2 = min(y2, img_meta['height'])

            MaskIm = cv2.resize(Mask, (int(x2-x1), int(y2-y1)), interpolation=cv2.INTER_NEAREST)  # integer mask 0..14
            MaskRdy = np.zeros((img_meta['height'], img_meta['width'], num_seg_patches), dtype=np.float32)
            MaskRdy[:, :, 0] = 1.  # bg outside the box
            MaskRdy[y1:y2, x1:x2, :] = to_categorical(  # inside the bbox
                MaskIm.astype(np.int), num_classes=num_seg_patches
            )
            # M.append(MaskRdy)
            M += MaskRdy

    # M = np.sum(M, axis=0).astype(np.float32)
    sum_person_mask = np.sum(M[:, :, 1:], axis=2)
    no_persons_mask = np.equal(sum_person_mask, 0.)
    sum_person_mask[sum_person_mask == 0.] = 1.  # to avoid dividing by 0.
    M[:, :, 1:] = M[:, :, 1:] / sum_person_mask[..., None]
    M[:, :, 0] = no_persons_mask.astype(np.float32)

    return M

def get_densepose_points(coco_obj, img_id, size=None):
    """Given a COCO dataset object, retrieve Densepose points as an array of shape (N, 5),
    where N is a number of points provided. Each point is a row of 5: vertical and horizontal coordinates,
    body patch index, u, v coordinates.
    
    # Returns: np.array of shape (N, 5) and float32 dtype."""
    img_meta = coco_obj.imgs[img_id]
    # load anns for that image
    anns = coco_obj.imgToAnns[img_id]

    YX = []
    IUV = []  
            
    for ann in anns:
        bbr =  np.array(ann['bbox']).astype(int)
        if 'dp_masks' in ann.keys():
            Point_x = np.array(ann['dp_x'], dtype=float) / 255. * bbr[2]  # Strech the points to current box.
            Point_y = np.array(ann['dp_y'], dtype=float) / 255. * bbr[3]  # Strech the points to current box.

            Point_I = np.array(ann['dp_I'], dtype=float)
            Point_U = np.array(ann['dp_U'], dtype=float)
            Point_V = np.array(ann['dp_V'], dtype=float)

            x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0]+bbr[2], bbr[1]+bbr[3]
            x2 = min(x2, img_meta['width'])
            y2 = min(y2, img_meta['height'])

            Point_x += x1
            Point_y += y1

            points_yx = np.concatenate([
                Point_y.reshape(-1, 1),
                Point_x.reshape(-1, 1)
            ], axis=1)

            points_iuv = np.concatenate([
                Point_I.reshape(-1, 1),
                Point_U.reshape(-1, 1),
                Point_V.reshape(-1, 1)
            ], axis=1)

            YX.append(points_yx)
            IUV.append(points_iuv)

    return np.concatenate([np.concatenate(YX, axis=0), np.concatenate(IUV, axis=0)], axis=1)


def get_denspose_part_segments_loss_region(dp_seg, bin_seg):
    """Get an intersection of binary mask and Densepose mask annotations.
    
    # Returns: np.array of shape (H, W) and float32 dtype."""
    return (~(dp_seg[:, :, 0].astype(np.bool)) == bin_seg.astype(np.bool)).astype(np.float32)


def generate_neg_coords(seg):
    """Generate a grid of points that belong to the background given segmentation annotation.
    First two columns are coordinates (vertical and horizontal), three other columns are all 0.
    
    # Returns: np.array of shape (N, 5) and float32 dtype. It is possible for N to be 0."""
    
    delta = min(seg.shape) // 12
    prod_idx = list((product(range(delta, seg.shape[0]-delta//2, delta), range(delta, seg.shape[1]-delta//2, delta))))
    nseg = ~seg
    neg_coords = np.array(list(filter(lambda x: nseg[x[0], x[1]] == 1, prod_idx)))
    return np.concatenate([neg_coords.reshape((-1, 2)), np.zeros((neg_coords.shape[0], 3), dtype=np.float32)], axis=1)

def get_instance_offsets(coco_obj, img_id, size=None):
    """Return instance masks of instances that posess head."""
    img_meta = coco_obj.imgs[img_id]
    all_anns = coco_obj.imgToAnns[img_id]
    instance_masks = np.zeros((img_meta['height'],  img_meta['width']), dtype=np.int)
    #loss_region = np.zeros((img_meta['height'],  img_meta['width']), dtype=np.bool)
    
    head_points = []
    inst_idx = 1
    for pann in all_anns:
        if pann['keypoints']:
            pts = keypoints2numpy(pann['keypoints'])
            # print(pts)
            # determine if there is a head
            # first 5 points are head, so if there is some non zero presence value => there is head
            head = pts[:5, 2].sum(axis=0) > 0
            
            rle = mask_util.frPyObjects(pann['segmentation'], img_meta['height'],  img_meta['width'])
            m = mask_util.decode(rle).astype(np.bool)
            m = np.logical_or.reduce(m, axis=2) if len(m.shape) == 3 else m
            
            if head:
                # calculate a head point
                head_point = pts[:5, :2][pts[:5, 2] != 0].mean(axis=0)
                # add mask of the person to loss region
                # loss_region |= m
                # ins_mask = 
                instance_masks[m] = inst_idx
                head_points.append(head_point)
                # calculate vectors from all pixels to the head point
                #xx, yy = np.where(m)
                #print(xx.shape, yy.shape)
                #offset_mask[xx, yy, 0] = head_point[0] - xx
                #offset_mask[xx, yy, 1] = head_point[1] - yy
                
                inst_idx += 1

    if size is not None:  # TODO
        pass
    
    if head_points:
        head_points = np.stack(head_points, axis=0)
    return instance_masks, head_points


def get_image_data(coco_obj, img_id, base_path, toggle_image=True,
                   toggle_bin_mask=True, toggle_joints=True, toggle_dp_seg=True, toggle_dp_points=True,
                  toggle_instance_offsets=True):
    """Given a DPCOCO object, image id and directory with images, collects selected labels in a dictionary.
    # Keys: ['image', 'joints', 'joints_loss_region', 'bin_mask', 'dp_mask', 'dp_mask_loss_region', 'dp_points']
    # Returns: a dictionary with things."""
    res= {}
    if toggle_image:
        res['image'] = get_image(coco_obj, img_id, base_path, grayscale=False, size=None)
    if toggle_joints:
        res['joints'] = get_all_keypoints(coco_obj, img_id, size=None)
        res['joints_loss_region'] = get_keypoint_loss_region(coco_obj, img_id, size=None)  # all pixels except masks of persons that have no keypoints
    if toggle_bin_mask:
        res['bin_mask'] = get_binary_seg(coco_obj, img_id, size=None)
    if toggle_dp_seg:
        res['dp_mask'] = get_densepose_part_segments(coco_obj, img_id, size=None)
        if toggle_bin_mask:
            res['dp_mask_loss_region'] = get_denspose_part_segments_loss_region(res['dp_mask'], res['bin_mask'])  # intersection of densepose labels and coco
    if toggle_dp_points:
        res['dp_points'] = get_densepose_points(coco_obj, img_id, size=None)
    if toggle_instance_offsets:
        res['instance_offsets'], res['head_points'] = get_instance_offsets(coco_obj, img_id, size=None)
    return res

def get_data_from_image_id(
        image_id, coco_obj, img_size, base_path, grayscale=False,
        generate_negative_points=True, aug_pipeline=None, 
        toggle_bin_mask=True,
        toggle_joints=True,
        toggle_dp_seg=True,
        toggle_dp_points=True,
        toggle_instance_offsets=True                      
    ):
    all_results = {}
    
    res = get_image_data(
        coco_obj,
        image_id,
        base_path,
        toggle_image=True,
        toggle_bin_mask=toggle_bin_mask,
        toggle_joints=toggle_joints,
        toggle_dp_seg=toggle_dp_seg,
        toggle_dp_points=toggle_dp_points,
        toggle_instance_offsets=toggle_instance_offsets
    )
    
    # image
    image = res['image']
    xyhw_box = (0, 0, image.shape[0], image.shape[1])  # y, x, h, w
    xyhw_box = bbu.extend_xyhw_to_ratio(xyhw_box, img_size[1]/img_size[0])
    xyhw_box = bbu.round_bbox_params(xyhw_box)
    padded_img = imu.pad_to_bbox(image, xyhw_box, mode='mean')
    new_bbox = (max(0, xyhw_box[0]), max(0, xyhw_box[1]), xyhw_box[2], xyhw_box[3])
    offset_x, offset_y = xyhw_box[0], xyhw_box[1]
    padded_crop = padded_img[new_bbox[0]:new_bbox[0]+new_bbox[2], new_bbox[1]:new_bbox[1]+new_bbox[3]]
    old_shape = padded_crop.shape[:]
    resized_crop = cv2.resize(padded_crop, img_size[::-1], interpolation=cv2.INTER_AREA)
    all_results['image'] = resized_crop
    
    # keypoints
    if toggle_joints:
        all_keypoints = res['joints'].copy()
        all_shifted_pts = all_keypoints - [offset_x, offset_y, 0, 0]  # coordinates in the crops frame
        all_shifted_pts = all_shifted_pts[  # remove ousiders
            (0 <= all_shifted_pts[:, 0]) &
            (all_shifted_pts[:, 0] < padded_crop.shape[0]) &
            (0 <= all_shifted_pts[:, 1]) &
            (all_shifted_pts[:, 1] < padded_crop.shape[1])
        ]

        # rescale keypoints
        all_rescaled_pts = all_shifted_pts * [img_size[0]/old_shape[0], img_size[1]/old_shape[1], 1., 1.]
        all_results['joints'] = all_rescaled_pts

        # fix joints loss region
        joints_loss_region = res['joints_loss_region']
        joints_loss_region = imu.pad_to_bbox(joints_loss_region, xyhw_box, mode='constant', cval=1)  # pad with 1. to collect loss from no mask outside the image
        joints_loss_region = joints_loss_region[new_bbox[0]:new_bbox[0]+new_bbox[2], new_bbox[1]:new_bbox[1]+new_bbox[3]]
        joints_loss_region = cv2.resize(joints_loss_region.astype(np.uint8), img_size[::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)
        all_results['joints_loss_region'] = grey_erosion(joints_loss_region, 5)
    
    # pad, crop and resize bin_mask
    if toggle_bin_mask:
        _bin_mask = res['bin_mask']
        _bin_mask = imu.pad_to_bbox(_bin_mask, xyhw_box, mode='constant')
        _bin_mask = _bin_mask[new_bbox[0]:new_bbox[0]+new_bbox[2], new_bbox[1]:new_bbox[1]+new_bbox[3]]
        bin_mask = cv2.resize(_bin_mask.astype(np.uint8), img_size[::-1], interpolation=cv2.INTER_NEAREST).astype(np.float32)
        all_results['bin_mask'] = bin_mask
    
    # pad, crop and resize dp_mask
    if toggle_dp_seg:
        _dp_mask = res['dp_mask']
        _dp_mask = imu.pad_to_bbox(_dp_mask, xyhw_box, mode='constant')
        _dp_mask[:, :, 0] = ~np.logical_or.reduce(_dp_mask[:, :, 1:], axis=2)
        _dp_mask = _dp_mask[new_bbox[0]:new_bbox[0]+new_bbox[2], new_bbox[1]:new_bbox[1]+new_bbox[3]]
        dp_mask = resize(_dp_mask.astype(np.float32), img_size, order=0, mode='edge', anti_aliasing=False).astype(np.float32)
        # dp_mask = softmax(dp_mask, axis=2)
        dp_mask /= dp_mask.sum(axis=2)[:, :, None]
        all_results['dp_mask'] = dp_mask
    
    # rescale densepose points
    if toggle_dp_points:
        _dp_points = res['dp_points'].copy()
        _dp_points = _dp_points - [offset_x, offset_y, 0., 0., 0.]  # coordinates in the crops frame
        _dp_points = _dp_points[  # remove ousiders
            (0 <= _dp_points[:, 0]) &
            (_dp_points[:, 0] < padded_crop.shape[0]) &
            (0 <= _dp_points[:, 1]) &
            (_dp_points[:, 1] < padded_crop.shape[1])
        ]
        # rescale keypoints
        dp_points = _dp_points * [img_size[0]/old_shape[0], img_size[1]/old_shape[1], 1., 1., 1.]
        if generate_negative_points:
            dp_points = np.concatenate([dp_points, generate_neg_coords(bin_mask.astype(np.bool) | (~dp_mask[:, :, 0].astype(np.bool)))], axis=0)
        all_results['dp_points'] = dp_points
    
    # instance offsets
    if toggle_instance_offsets:
        inst_offsets, head_points = res['instance_offsets'], res['head_points']
        inst_offsets = imu.pad_to_bbox(inst_offsets, xyhw_box, mode='constant')
        inst_offsets = inst_offsets[new_bbox[0]:new_bbox[0]+new_bbox[2], new_bbox[1]:new_bbox[1]+new_bbox[3]]
        # inst_offsets /= list(inst_offsets.shape[:2])  # normalize offsets
        inst_offsets = cv2.resize(inst_offsets, img_size[::-1], interpolation=cv2.INTER_NEAREST)
        all_results['instance_offsets'] = inst_offsets

        if head_points != []:
            head_points = head_points - [offset_x, offset_y]
            head_points = head_points * [img_size[0]/old_shape[0], img_size[1]/old_shape[1]]
        all_results['head_points'] = head_points
    
    ################## AUGMENTS HERE ###################
    if aug_pipeline is not None:
        all_results = aug_pipeline(all_results)
    
    if toggle_instance_offsets:
        pts = all_results['head_points']
        all_results['instance_offsets_loss_region'] = (all_results['instance_offsets'] > 0).astype(np.float32)
        offset_mask = np.zeros((img_size[0], img_size[1], 2), dtype=np.float32)
        if pts != []:
            for i in range(pts.shape[0]):
                head_point = pts[i]
                # convert instance maps into offset maps and loss region
                xx, yy = np.where(all_results['instance_offsets'] == i + 1)

                offset_mask[xx, yy, 0] = head_point[0] - xx
                offset_mask[xx, yy, 1] = head_point[1] - yy
        all_results['instance_offsets'] = offset_mask / [img_size[0], img_size[1]]

        del all_results['head_points']
    return all_results


_affine_params = {
    "scale": (0.8, 1.4),
    "translate_percent": {'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
    "rotate": (-35, 35),
    "shear": (-4, 4),
}


def get_augmenter_images():
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.75,
            iaa.Affine(
                **_affine_params,
                order=[1, 3],
                cval=(100, 200),
                mode="constant",
                name="MyAffine"
            ),
            name="Sometimes/Affine"
        ),
        iaa.Sometimes(0.2, iaa.GaussianBlur((0.1, 2.), name="MyBlur"), name="Sometimes/Blur"),
        iaa.Add(value=(-10, 10), per_channel=True, name="MyAdd"),
    ], name="MySeq")
    return seq


def get_augmenter_geometry():
    seq = iaa.Sequential([
        iaa.Sometimes(
            0.75,
            iaa.Affine(
                **_affine_params,
                order=0,
                cval=0,
                mode="const",
                name="MyAffine"
            ),
            name="Sometimes/Affine"
        ),
    ], name="MySeq")
    return seq

aug_pipeline_geometry = get_augmenter_geometry()
aug_pipeline_image = get_augmenter_images()

def augment_data(
        data_dict,
        aug_pipeline_geom,
        aug_pipeline_image,
        toggle_bin_mask=True,
        toggle_joints=True,
        toggle_dp_seg=True,
        toggle_dp_points=True,
        toggle_instance_offsets=True
    ):
    
    transformed = {}
    
    seq_img_i = aug_pipeline_image.to_deterministic()
    seq_img_i = seq_img_i.localize_random_state()
    seq_masks_i = aug_pipeline_geom.to_deterministic()
    seq_masks_i = seq_masks_i.copy_random_state(seq_img_i, matching="name")

    # augment image
    transformed["image"] = seq_img_i.augment_image(data_dict["image"])
    im_size = data_dict["image"].shape[:2]
    
    # augment densepose maps
    if toggle_dp_seg:
        mask = ia.SegmentationMapOnImage(data_dict["dp_mask"][:, :, 1:], im_size, nb_classes=None)
        mask = seq_masks_i.augment_segmentation_maps(mask)
        mask = mask.arr
        mask = np.concatenate([1 - mask.sum(axis=2, keepdims=True), mask], axis=2)
        transformed["dp_mask"] = mask 
        if mask.sum(axis=2).max() > 1.:
            raise ValueError("What a shitshow!")
    
    # augment binary masks
    if toggle_bin_mask:
        bin_mask = ia.SegmentationMapOnImage(data_dict["bin_mask"][:, :, None], im_size, nb_classes=1)
        bin_mask = seq_masks_i.augment_segmentation_maps(bin_mask)
        transformed["bin_mask"] = bin_mask.arr[:, :, 0]
    
    # augment joint loss region
    if toggle_joints:
        jlr = ia.SegmentationMapOnImage(1. - data_dict["joints_loss_region"][:, :, None], im_size, nb_classes=1)
        jlr = seq_masks_i.augment_segmentation_maps(jlr)
        transformed["joints_loss_region"] = 1. - jlr.arr[:, :, 0]
    
        # augment joints
        joints = ia.KeypointsOnImage.from_coords_array(data_dict["joints"][:, [1, 0]], shape=im_size)
        joints = seq_img_i.augment_keypoints(joints)
        joints = joints.get_coords_array()[:, [1, 0]]
        joints = np.concatenate([joints, data_dict["joints"][:, 2:]], axis=1)
        # transformed["all_joints"] = joints.copy()
        # filter outsider joints
        joints = joints[
            (0 <= joints[:, 0]) &
            (joints[:, 0] < im_size[0]) &
            (0 <= joints[:, 1]) &
            (joints[:, 1] < im_size[1])
        ]
        transformed["joints"] = joints
    
    # augment densepose points
    if toggle_dp_points:
        dp_points = ia.KeypointsOnImage.from_coords_array(data_dict["dp_points"][:, [1, 0]], shape=data_dict["image"].shape[:2])
        dp_points = seq_img_i.augment_keypoints(dp_points)
        dp_points = dp_points.get_coords_array()[:, [1, 0]]
        dp_points = np.concatenate([dp_points, data_dict["dp_points"][:, 2:]], axis=1)
        # filter outsider joints
        dp_points = dp_points[
            (0 <= dp_points[:, 0]) &
            (dp_points[:, 0] < im_size[0]) &
            (0 <= dp_points[:, 1]) &
            (dp_points[:, 1] < im_size[1])
        ]
        transformed["dp_points"] = dp_points

    # augment instance masks for instance offsets
    if toggle_instance_offsets:
        nb_classes=data_dict["instance_offsets"].max()+1
        inst_off_masks = ia.SegmentationMapOnImage(data_dict["instance_offsets"], im_size,
                                                   nb_classes=nb_classes)
        inst_off_masks = seq_masks_i.augment_segmentation_maps(inst_off_masks)
        #print(inst_off_masks.arr.shape, inst_off_masks.arr.dtype, inst_off_masks.arr.max())
        transformed["instance_offsets"] = (inst_off_masks.arr @ np.arange(nb_classes)).astype(np.int)

        # augment head points
        head_points = data_dict["head_points"]
        if head_points != []:
            head_points = ia.KeypointsOnImage.from_coords_array(data_dict["head_points"][:, [1, 0]], shape=im_size)
            head_points = seq_img_i.augment_keypoints(head_points)
            head_points = head_points.get_coords_array()[:, [1, 0]]
        transformed["head_points"] = head_points
    
    return transformed

def aug_pipeline(
        data_dict,
        toggle_bin_mask=True,
        toggle_joints=True,
        toggle_dp_seg=True,
        toggle_dp_points=True,
        toggle_instance_offsets=True        
    ):
    return augment_data(
        data_dict,
        aug_pipeline_geometry,
        aug_pipeline_image,
        toggle_bin_mask=toggle_bin_mask,
        toggle_joints=toggle_joints,
        toggle_dp_seg=toggle_dp_seg,
        toggle_dp_points=toggle_dp_points,
        toggle_instance_offsets=toggle_instance_offsets
    )


class RevBatchGen(Sequence):
    """A Data Generator for DensePose COCO."""
    def __init__(
        self,
        image_anns_keys,
        coco_obj,
        batch_size,
        base_path,
        image_size=(224, 224),
        gt_size=(224, 224),
        shuffle=False,
        augment=False,
        grayscale=False,
        joints_sigma=2.,
        preprocessing_function=None,
        keypoint_magnitude_factor=1.,
        augment_pipeline=None,
        bce_approach=True,
        generate_neg_points=True,
        toggle_bin_mask=True,
        toggle_joints=True,
        toggle_dp_seg=True,
        toggle_dp_points=True,
        toggle_instance_offsets=True
    ):
        self.toggle_bin_mask = toggle_bin_mask
        self.toggle_joints = toggle_joints
        self.toggle_dp_seg = toggle_dp_seg
        self.toggle_dp_points = toggle_dp_points
        self.toggle_instance_offsets = toggle_instance_offsets
        
        self.generate_neg_points = generate_neg_points
        if preprocessing_function is None:
            self.preprocessing_function = lambda x: x.astype(np.float32)
        else:
            self.preprocessing_function = preprocessing_function
        self.keypoint_magnitude_factor = keypoint_magnitude_factor  # magically improves convergence
        self.joints_sigma = joints_sigma
        self.grayscale = grayscale
        self.base_path = base_path
        self.batch_size = batch_size
        # self.augment = augment
        self.image_size = image_size
        self.gt_size = gt_size
        self.shuffle = shuffle
        self.augment = augment
        self.coco_obj = coco_obj
        self.indices = np.array(image_anns_keys)
        self.bce_approach = bce_approach
        
        if augment_pipeline is not None:
            self.augment_pipeline = lambda data_dict: augment_pipeline(
                data_dict,
                toggle_bin_mask=toggle_bin_mask,
                toggle_joints=toggle_joints,
                toggle_dp_seg=toggle_dp_seg,
                toggle_dp_points=toggle_dp_points,
                toggle_instance_offsets=toggle_instance_offsets
            )
        else:
            self.augment_pipeline = None
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx*(self.batch_size):(idx + 1)*self.batch_size]
        
        batch_list = []
        for bidx in batch_indices:

            res_dict = get_data_from_image_id(
                bidx,
                self.coco_obj,
                self.image_size,
                self.base_path,
                grayscale=self.grayscale,
                generate_negative_points=self.generate_neg_points,
                aug_pipeline=self.augment_pipeline,
                toggle_bin_mask=self.toggle_bin_mask,
                toggle_joints=self.toggle_joints,
                toggle_dp_seg=self.toggle_dp_seg,
                toggle_dp_points=self.toggle_dp_points,
                toggle_instance_offsets=self.toggle_instance_offsets
            )
            
            # preprocess image
            res_dict['image'] = self.preprocessing_function(res_dict['image'])
            
            if self.toggle_bin_mask:
                res_dict['bin_mask'] = resize(
                    res_dict['bin_mask'].astype(np.float32),
                    self.gt_size,
                    order=0, mode='edge', anti_aliasing=False
                ).astype(np.float32)[:, :, None]
            
            #res_dict['joints_loss_region'] = res_dict['joints_loss_region'].astype(np.float32)[:, :, None]
            if self.toggle_joints:
                res_dict['joints_loss_region'] = resize(
                    res_dict['joints_loss_region'].astype(np.float32),
                    self.gt_size,
                    order=0, mode='edge', anti_aliasing=False
                ).astype(np.float32)[:, :, None]
            
                # draw heatmap
                if not self.bce_approach:
                    res_dict['joints'] = points_to_heatmap(res_dict['joints'], self.image_size, self.gt_size, self.joints_sigma) * self.keypoint_magnitude_factor
                else:
                    res_dict['joints'] = points_to_heatmap_bce_approach(res_dict['joints'], self.image_size, self.gt_size, self.joints_sigma)
            
            if self.toggle_instance_offsets:
                res_dict['instance_offsets_loss_region'] = res_dict['instance_offsets_loss_region'].astype(np.float32)[:, :, None]
            
            # rescale dp_mask
            if self.toggle_dp_seg:
                res_dict['dp_mask'] = resize(res_dict['dp_mask'].astype(np.float32), self.gt_size, order=0, mode='edge', anti_aliasing=False).astype(np.float32)
                res_dict['dp_mask'] /= res_dict['dp_mask'].sum(axis=2)[:, :, None]
            
            # draw densepose points
            if self.toggle_dp_points:
                dp_points = draw_dp_points_on_canvas(res_dict['dp_points'], self.image_size, desired_size=self.gt_size)

                if not self.generate_neg_points:
                    bg_mask = np.logical_and(res_dict['bin_mask'] == 0, res_dict['dp_mask'][:, :, [0]] == 1)
                    dp_points *= np.logical_not(bg_mask)  # False is bg
                    dp_points[:, :, 0] = bg_mask.astype(np.float32)[:, :, 0]

                res_dict['dp_points'] = dp_points
            
            batch_list.append(res_dict)
        
        # {print(batch_list[i][key].shape, key, i) for i in range(self.batch_size) for key in batch_list[0].keys()}        
        
        X = {key: np.stack([batch_list[i][key] for i in range(self.batch_size)], axis=0) for key in batch_list[0].keys()}
        return (X, None)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def draw_dp_points_on_canvas(dp_points, initial_size, desired_size=None):
    """Takes an array of densepose points of shape (N, 5): y, x, part, u, v.
    And draws them as pixels.
    
    # Returns: np.array of shape `(*desired_size, 25+24+24)`.
    First 25 channels used for semantic segmentation of patches.
    Other 24+24 channels are for UV coordinates regression."""
    if desired_size is None:
        desired_size = initial_size
        
    field = np.zeros((*desired_size, 25+24+24), dtype=np.float32)  # categorical 25 for patches + bg, 24 for u and v
    
    # rescale dp 
    dp_points = dp_points * [desired_size[0]/initial_size[0], desired_size[1]/initial_size[1], 1., 1., 1.]
    for i in range(dp_points.shape[0]):
        y, x, part, u, v = dp_points[i, :]
        y, x, part = min(int(round(y)), desired_size[0]-1), min(int(round(x)), desired_size[1]-1), int(part)
        # assert part >= 0 and part <= 24
        field[y, x, ...] = 0.
        field[y, x, part] = 1.
        if part != 0:
            field[y, x, part + 24] = u 
            field[y, x, part + 48] = v
    # assert field[:, :, :25].sum(axis=2).max() == 1.
    return field


def get_filtered_person_dict(coco_obj):
    """Crutchy little function to filter person ids and select only good ones for training.
    
    # Returns: a dict with DPCOCO person annotaions."""
    filtered_person_anns = {}
    for pers_id, pers_ann in coco_obj.anns.items():
        # skip crowd labels
        if pers_ann['iscrowd']:
            continue

        # skip persons without densepose lables
        if not bool(pers_ann.get('dp_masks')):
            continue       
        
        # if the image contains persons without keypoints -> continue | no need for that as we employ loss mask for joints
        # if any((False if (ann['num_keypoints'] != 0) else True) for ann in coco_obj.imgToAnns[pers_ann['image_id']]):
        #     continue

        # select large enough persons
        if (
            ((pers_ann['bbox'][2] >= 90) or (pers_ann['bbox'][3] >= 90)) and (pers_ann['num_keypoints'] >= 5)
        ):  # coco bbox (x, y, w, h)
            filtered_person_anns[pers_id] = pers_ann
    return filtered_person_anns


class UnFreezer(keras.callbacks.Callback):
    def __init__(self, lr=None, unfreeze_after=0):
        self.lr = lr
        self.unfreeze_after = unfreeze_after
    def on_epoch_end(self, epoch, logs={}):
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        if epoch == self.unfreeze_after:
            print("#################### UNFREEZING ####################")
            for layer in self.model.layers:
                layer.trainable = True
            K.set_value(self.model.optimizer.lr, self.lr)
            self.model.compile(self.model.optimizer, None, None)
