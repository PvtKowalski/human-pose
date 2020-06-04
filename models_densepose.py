import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
from models_fbp import get_FBP_models, get_dilated_unet, conv_block


def get_DUV_model(
        input_shape=(224, 224, 3),
        kernel_regularizer=None,
        weights='imagenet',
        densepose_parts=15,
        densepose_patches=24,
        num_joints=17,
        base_last_feats=384,
        checkpoint_FBP=None,
        skip_names_FBP=('pyramid_relu', 'B_2_reduced_act', 'P_2_reduced_act'),  # 'conv2d_21', 'conv2d_44'
        with_loss=True
    ):

    model, _ = get_FBP_models(
        input_shape=input_shape,
        kernel_regularizer=kernel_regularizer,
        weights=weights,
        densepose_parts=densepose_parts,
        densepose_patches=densepose_patches,
        num_joints=num_joints,
        base_last_feats=base_last_feats,
        with_loss=False
    )
    # load pretrained segmentation weigths
    if checkpoint_FBP is not None:
        model.load_weights(checkpoint_FBP)
        print("########### SEGMENTATION CHECKPOINT LOADED ###############")
    
    freeze_layers = model.layers
    
    F, B, P = (
        model.get_layer(name=skip_names_FBP[0]).output,
        model.get_layer(name=skip_names_FBP[1]).output,
        model.get_layer(name=skip_names_FBP[2]).output,
    )
    
    x = keras.layers.Concatenate(name='concatenate_FBP')([F, B, P])
    
    # split parts seg and UV
    D = get_dilated_unet(
        x,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=64,#densepose_patches+1,  # num patches + bg
        bottleneck_depth=4,
        filters_bottleneck=192,
        last_name='D_2_reduced'
    )
    D = keras.layers.ReLU(name='D_2_reduced_act')(D)
    
    # high res patches
    D_out = keras.layers.UpSampling2D()(D)
    D_out = conv_block(  # conv to smooth things out
        D_out, 32, kernel_size=(5, 5),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    D_out = keras.layers.Conv2D(
        densepose_patches+1,
        (1, 1),
        activation=None,
        name='pred_D',
        kernel_regularizer=kernel_regularizer
    )(D_out)
    
    x1 = keras.layers.Concatenate(name='concatenate_FD')([F, D])
    
    # UV subnetwork
    UV = get_dilated_unet(
        x1,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=64,#2*densepose_patches,  # 24 for U, 24 for V
        bottleneck_depth=4,
        last_name='UV_2_reduced',
        decoder_filters=192  # fat convs for decoder to carry UV info 
    )
    UV = keras.layers.ReLU(name='UV_2_reduced_act')(UV)
    
    # high res UV output
    UV_out = keras.layers.UpSampling2D()(UV)
    UV_out = conv_block(  # conv to smooth things out
        UV_out, 64, kernel_size=(5, 5),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    UV_out = keras.layers.Conv2D(
        2*densepose_patches,
        (1, 1),
        activation=None,
        name='pred_UV',
        kernel_regularizer=kernel_regularizer
    )(UV_out)
    
    DUV = keras.layers.Concatenate(name='output_densepose_uv')([D_out, UV_out])
    
    #input_binary_mask = keras.layers.Input(shape=(*gt_size, 1), name='bin_mask')
    #input_densepose_mask = keras.layers.Input(shape=(*gt_size, densepose_parts), name='dp_mask')
    input_densepose_uv = keras.layers.Input(shape=(input_shape[0], input_shape[1], densepose_patches*3 + 1), name='dp_points')
    
    if with_loss:
        loss = compute_losses_DUV(
            [*model.outputs, DUV],
            [model.inputs[1], model.inputs[2], input_densepose_uv]  # input_binary_mask, input_densepose_mask, 
        )
    # input_binary_mask, input_densepose_mask, 
    duv_mdl = keras.models.Model([*model.inputs, input_densepose_uv], [*model.outputs, DUV])
    if with_loss:
        duv_mdl.add_loss(loss)
    return duv_mdl, freeze_layers


def logcosh(x):
    """Computes a logarithm of hyperbolic cosine with keras backend."""
    return x + K.softplus(-2. * x) - K.log(2.)


def compute_losses_DUV(preds, truths):
    """Festival of crutches."""
    bin_mask_true, dp_mask_true, dp_uv_true = truths
    bin_mask_pred, dp_mask_pred, dp_uv_pred = preds
    w2, w3, w4, w5, w6 = (2., 2., 20., 1.5e4, 1.5e4)#(.5, .5, 10., 1e4, 1e4)
    weight_for_bg = 1./1200.
    
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
        w6 * dp_v_loss
    )
