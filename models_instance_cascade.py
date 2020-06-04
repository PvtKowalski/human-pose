import keras
import keras.backend as K
import tensorflow as tf

import models_densepose
import models_fbp
import numpy as np
from models_fbp import conv_block


def get_full_cascade_model(
        input_shape=(224, 224, 3),
        kernel_regularizer=None,
        weights='imagenet',
        gt_size=(112, 112), 
        densepose_parts=15,
        densepose_patches=24,
        num_joints=17,
        base_last_feats=384,
        checkpoint_DUV=None,
        skip_names_DUV=(
            'pyramid_relu',
            'B_2_reduced_act',
            'P_2_reduced_act',
            'D_2_reduced_act',
            'UV_2_reduced_act',
        ),  # 'conv2d_21', 'conv2d_44'
        with_loss=True
    ):
    
    mdl, _ = models_densepose.get_DUV_model(
        input_shape=input_shape,
        kernel_regularizer=kernel_regularizer,
        weights=weights,
        densepose_parts=densepose_parts,
        densepose_patches=densepose_patches,
        num_joints=num_joints,
        base_last_feats=base_last_feats,
        checkpoint_FBP=None,
        skip_names_FBP=skip_names_DUV[:3],
        with_loss=False
    )
    
    if checkpoint_DUV is not None:
        mdl.load_weights(checkpoint_DUV)
    
    F, B, P, D, UV = [mdl.get_layer(nm).output for nm in skip_names_DUV]
    
    x = keras.layers.Concatenate(name='concat_FP')([F, P])
    # obtain keypoints from base feature pyramid and parts
    K = models_fbp.get_dilated_unet(
        x,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=32,#num_joints,  # coco joints
        bottleneck_depth=3,
        last_name='K_2_reduced',  # 'pred_K'
    )
    K = keras.layers.ReLU(name='K_2_reduced_act')(K)
    
    # high res patches
    K_out = keras.layers.UpSampling2D()(K)
    K_out = conv_block(  # conv to smooth things out
        K_out, 32, kernel_size=(5, 5),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    K_out = keras.layers.Conv2D(
        num_joints,
        (1, 1),
        activation=None,
        name='output_joints',
        kernel_regularizer=kernel_regularizer
    )(K_out)
    
    x1 = keras.layers.Concatenate(name='concat_FPDUVK')([
        F,
        P,
        D,
        UV,
        K
    ])
    # mix almost all features for hough voting
    H = models_fbp.get_dilated_unet(
        x1,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=128,  # 24 for U, 24 for V
        bottleneck_depth=4,
        last_name='H_2_reduced',
        decoder_filters=128  # fat convs for decoder to carry Hough voting info 
    )
    H = keras.layers.ReLU(name='H_2_reduced_act')(H)
    # high res patches
    H_out = keras.layers.UpSampling2D()(H)
    H_out = conv_block(  # conv to smooth things out
        H_out, 32, kernel_size=(3, 3),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    H_out = keras.layers.Conv2D(
        2,
        (1, 1),
        activation=None,
        name='output_instance_offsets',
        kernel_regularizer=kernel_regularizer
    )(H_out)

    input_keypoints = keras.layers.Input(shape=(*input_shape[:2], num_joints), name='joints')
    input_keypoints_loss_region = keras.layers.Input(shape=(*input_shape[:2], 1), name='joints_loss_region')
    input_instance_offsets = keras.layers.Input(shape=(*input_shape[:2], 2), name='instance_offsets')
    input_io_loss_region = keras.layers.Input(shape=(*input_shape[:2], 1), name='instance_offsets_loss_region')
    
    loss = compute_all_losses(
        [*mdl.outputs, K_out, H_out],
        [
            mdl.inputs[1], mdl.inputs[2], mdl.inputs[3],
            input_keypoints_loss_region, input_keypoints, 
            input_io_loss_region, input_instance_offsets
        ]
    )
    
    a_model = keras.models.Model(
        inputs=[
            *mdl.inputs,
            input_keypoints_loss_region,
            input_keypoints,
            input_io_loss_region,
            input_instance_offsets
        ],
        outputs=[*mdl.outputs, K_out, H_out]
    )
    a_model.add_loss(loss)
    return a_model, mdl.layers


def logcosh(x):
    """Computes a logarithm of hyperbolic cosine with keras backend."""
    return x + K.softplus(-2. * x) - K.log(2.)

def compute_all_losses(preds, truths):
    """Festival of crutches."""
    bin_mask_true, dp_mask_true, dp_uv_true, kpts_loss_mask, kpts_true, io_mask, inst_offsets_true = truths
    bin_mask_pred, dp_mask_pred, dp_uv_pred, kpts_pred, inst_offsets = preds
    w2, w3, w4, w5, w6 = (2., 2., 20., 1.5e4, 1.5e4)#(.5, .5, 10., 1e4, 1e4)
    w7, w8 = 0.5, 1e2  # 2., 1e3.
    weight_for_bg = 1./1200.
    
    Y_map = np.tile(np.linspace(0.5, -0.5, 256, dtype=np.float32), [256, 1])  # TODO: hardcoded shit
    X_map = Y_map.T
    # 2d vector field that points to the middle: top-left value is (0.5, 0.5), bot-right is (-0.5, -0.5) 
    XY_map = K.constant(np.stack([X_map, Y_map], axis=-1)[None, :, :, :], dtype=K.floatx())
    
    # instance offsets
    # we add our constant vector map, so that net has to predict a constanct distinct vector for each instance
    # jesus please make it work
    io_loss = K.mean(logcosh((inst_offsets + XY_map) - inst_offsets_true) * io_mask)
    
    # keypoints loss bce
    kpts_loss = K.mean(K.binary_crossentropy(kpts_true, kpts_pred, from_logits=True) * kpts_loss_mask)
    
    # intersection of binary and body part masks
    seg_loss_mask = K.cast(
        K.equal(1 - bin_mask_true, K.expand_dims(dp_mask_true[..., 0], axis=-1)), dtype=K.floatx()
    )

    # binmask loss
    bin_mask_loss = K.mean(K.binary_crossentropy(bin_mask_true, bin_mask_pred, from_logits=True))
    
    # densepose mask loss
    dp_mask_loss = K.mean(
        K.expand_dims(
            tf.nn.softmax_cross_entropy_with_logits_v2(dp_mask_true, dp_mask_pred, axis=-1),
            axis=-1
        ) * seg_loss_mask
    )
                                                                                                                         
    # densepose uv + weigh bg class way down
    dp_loss_mask = K.expand_dims(K.sum(dp_uv_true[..., :25], axis=-1), axis=-1)
    dp_bg_w_mask = (
        K.expand_dims(dp_uv_true[..., 0], axis=-1) * weight_for_bg + # weight for bg points
        K.expand_dims(1. - dp_uv_true[..., 0], axis=-1)  # 1 for rest...
    )
    
    # so that true labels are correct distribution (sum to 1, axis=-1) 
    crutch = 1. - dp_loss_mask
    dp_uv_true = K.concatenate([
        K.expand_dims(dp_uv_true[..., 0], axis=-1) + crutch,
        dp_uv_true[..., 1:]
    ], axis=-1) 
    
    cce = tf.nn.softmax_cross_entropy_with_logits_v2(dp_uv_true[..., :25], dp_uv_pred[..., :25], axis=-1)
    cce =  K.expand_dims(cce, axis=-1) * dp_loss_mask  # (?, 112, 112, 1)
    dp_patch_loss = K.mean(
         cce * dp_bg_w_mask,  # don't include loss from empty pixels and also balance bg loss way down
    )
    
    dp_u_loss = K.mean(logcosh(dp_uv_true[..., 25:25+24] - dp_uv_pred[..., 25:25+24]) * dp_uv_true[..., 1:25])
    dp_v_loss = K.mean(logcosh(dp_uv_true[..., 25+24:25+48] - dp_uv_pred[..., 25+24:25+48]) * dp_uv_true[..., 1:25])
    
    return (
        w2 * bin_mask_loss +
        w3 * dp_mask_loss +
        w4 * dp_patch_loss +
        w5 * dp_u_loss +
        w6 * dp_v_loss +
        w7 * kpts_loss +
        w8 * io_loss
    )
