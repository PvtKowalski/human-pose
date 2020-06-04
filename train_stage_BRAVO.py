import os
import glob
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append(os.path.abspath("../.."))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import keras
import keras.backend as K
import tensorflow as tf
from pycocotools import coco
import pycocotools.mask as mask_util

from common_utils import RevBatchGen, get_filtered_person_dict, aug_pipeline, UnFreezer


if __name__ == '__main__':
    # filter indices
    # leave only person ids that have densepose labels
    person_keypoints_train2014 = "/home/kowalski/datasets/mscoco2014/DensePoseData/DensePose_COCO/densepose_coco_2014_train.json"
    person_keypoints_val2014 = "/home/kowalski/datasets/mscoco2014/DensePoseData/DensePose_COCO/densepose_coco_2014_minival.json"

    coco_train = coco.COCO(person_keypoints_train2014)
    coco_val = coco.COCO(person_keypoints_val2014)

    filtered_person_anns_train = get_filtered_person_dict(coco_train)
    filtered_person_anns_val = get_filtered_person_dict(coco_val)
    filtered_image_keys_train = list({v['image_id'] for k, v in filtered_person_anns_train.items()})
    filtered_image_keys_val = list({v['image_id'] for k, v in filtered_person_anns_val.items()})
    print(
        len(coco_train.anns),
        len(coco_val.anns),
        len(filtered_image_keys_train),
        len(filtered_image_keys_val)
    )

    batch_size = 8
    base_path_train = "/home/kowalski/datasets/mscoco2014/train2014/"
    base_path_val = "/home/kowalski/datasets/mscoco2014/val2014/"
    image_size = (256, 256)
    gt_size = (256, 256)  # TODO: fix generator somehow to allow different sizes?
    grayscale = False
    bbox_expand_coeff = 1.3
    joints_sigma = 6.
    keypoint_magnitude_factor = 1.
    bce_approach = True
    preprocessing_tf = lambda x: x.astype(np.float32) / 127.5 - 1.
    toggle_bin_mask = True
    toggle_joints = False
    toggle_dp_seg = True
    toggle_dp_points = True
    toggle_instance_offsets = False

    gen_val = RevBatchGen(
        filtered_image_keys_val,
        coco_val,
        batch_size=batch_size,
        base_path=base_path_val,
        image_size=image_size,
        gt_size=gt_size,
        shuffle=False,
        augment=False,
        augment_pipeline=None,
        grayscale=grayscale,
        joints_sigma=joints_sigma,
        preprocessing_function=preprocessing_tf,
        keypoint_magnitude_factor=keypoint_magnitude_factor,
        bce_approach=bce_approach,
        generate_neg_points=False,
        toggle_bin_mask=toggle_bin_mask,
        toggle_joints=toggle_joints,
        toggle_dp_seg=toggle_dp_seg,
        toggle_dp_points=toggle_dp_points,
        toggle_instance_offsets=toggle_instance_offsets
    )

    gen_train = RevBatchGen(
        filtered_image_keys_train,
        coco_train,
        batch_size=batch_size,
        base_path=base_path_train,
        image_size=image_size,
        gt_size=gt_size,
        shuffle=True,
        augment=True,
        augment_pipeline=aug_pipeline,
        grayscale=grayscale,
        joints_sigma=joints_sigma,
        preprocessing_function=preprocessing_tf,
        keypoint_magnitude_factor=keypoint_magnitude_factor,
        bce_approach=bce_approach,
        generate_neg_points=False,
        toggle_bin_mask=toggle_bin_mask,
        toggle_joints=toggle_joints,
        toggle_dp_seg=toggle_dp_seg, 
        toggle_dp_points=toggle_dp_points,
        toggle_instance_offsets=toggle_instance_offsets
    )

    X, _ = gen_train[0]
    print("##########################################################")
    print(X['image'].shape, X['bin_mask'].shape, X['dp_mask'].shape, X['dp_points'].shape)
    print(X.keys())
    
    from models_densepose import get_DUV_model
    model, layers_to_freeze = get_DUV_model(
        input_shape=(*image_size, 3),
        kernel_regularizer=None,
        weights='imagenet',
        densepose_parts=15,
        densepose_patches=24,
        num_joints=17,
        base_last_feats=384,
        checkpoint_FBP="ALPHA_v1/model_ALPHA_022_0.1875.h5",
        skip_names_FBP=('pyramid_relu', 'B_2_reduced_act', 'P_2_reduced_act')  # 'conv2d_21', 'conv2d_44'
    )

    with open('BRAVO/summary_BRAVO.txt', 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    print(model.inputs)

    for l in layers_to_freeze:
        l.trainable = False
    
    base_lr = 0.0001
    optim = keras.optimizers.RMSprop(base_lr, clipnorm=2.5)
    model.compile(optim)

    def schedule(ep, lr):
        if ep <= 7:
            return base_lr
        elif ep <= 15:
            return base_lr / 5
        else:
            return base_lr / 25

    callbacks = [
        UnFreezer(base_lr, 3),
        keras.callbacks.CSVLogger('BRAVO/logs_BRAVO.csv', append=False),
        keras.callbacks.ModelCheckpoint(
            'BRAVO/model_BRAVO_{epoch:03d}_{val_loss:.4f}.h5',
            verbose=1,
            save_best_only=False,
            save_weights_only=False
        ),
        keras.callbacks.LearningRateScheduler(schedule, verbose=1),
    ]

    model.fit_generator(
        generator=gen_train,
        steps_per_epoch=None, ##
        epochs=2000,
        verbose=1,
        callbacks=callbacks,
        validation_data=gen_val,
        validation_steps=None, ##
        class_weight=None,
        max_queue_size=10,
        workers=2,
        use_multiprocessing=False,
        shuffle=False,
        initial_epoch=0 ##
    )
