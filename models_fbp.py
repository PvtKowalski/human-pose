import keras
import keras.backend as K
import tensorflow as tf


def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), kernel_regularizer=None, padding='same', activation='relu'):
    """Conv2D -> BN -> (ReLU)."""
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        strides=strides,
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_regularizer=kernel_regularizer
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    if activation is not None:
        x = keras.layers.Activation(activation)(x)
    return x


def create_poor_mans_fpn(
        input_shape=(384, 384, 3),
        alpha=1.4,
        kernel_regularizer=None,
        skip_squeeze=(32, 64, 128, 192, 256),
        last_feats=384
    ):

    inp = keras.layers.Input(shape=input_shape, name='image')
    mdl = keras.applications.mobilenet_v2.MobileNetV2(
        input_tensor=inp,
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )

    feature_pyramid_names = [
        # 'image',
        'expanded_conv_project_BN',
        'block_2_add',
        'block_5_add',
        'block_12_add',
        'block_16_project_BN',
    ]

    res = []
    for i, fn in enumerate(feature_pyramid_names):
        x = mdl.get_layer(name=fn).output
        if i == 0:
            x = conv_block(
                x, 16, kernel_size=(5, 5),
                kernel_regularizer=kernel_regularizer,
                padding='same', activation='relu')
            x = conv_block(
                x, 16, kernel_size=(5, 5),
                kernel_regularizer=kernel_regularizer,
                padding='same', activation='relu')
            x = conv_block(
                x, skip_squeeze[i], kernel_size=(1, 1),
                kernel_regularizer=kernel_regularizer,
                padding='same', activation='relu')
        elif i > 0:
            # skip squeeze
            x = conv_block(
                x, skip_squeeze[i], kernel_size=(1, 1),
                kernel_regularizer=kernel_regularizer,
                padding='same', activation='relu')
            x = keras.layers.UpSampling2D(
                size=(2**i, 2**i),
                interpolation='nearest',
                name=f'pyramid_upsample_{fn}'
            )(x)
        res.append(x)
    res = keras.layers.Concatenate(name='pyramid_concat')(res)
    # squeeze features to preserve compute and memory
    res = keras.layers.Conv2D(
        last_feats,
        (1, 1),
        kernel_regularizer=kernel_regularizer,
        name='pyramid_squeeze'
    )(res)  
    res = keras.layers.BatchNormalization(name='pyramid_bn')(res)
    res = keras.layers.LeakyReLU(alpha=0.05, name='pyramid_relu')(res)

    return keras.models.Model(inputs=mdl.inputs, outputs=[res], name='poor_mans_fpn'), mdl.layers


def encoder(x, filters=44, n_block=3, kernel_size=(3, 3), activation='relu', kernel_regularizer=None):
    skip = []
    for i in range(n_block):
        x = keras.layers.Conv2D(
            filters * 2**i,
            kernel_size,
            activation=activation,
            padding='same',
            kernel_regularizer=kernel_regularizer
        )(x)
        x = keras.layers.Conv2D(
            filters * 2**i,
            kernel_size,
            activation=activation,
            padding='same',
            kernel_regularizer=kernel_regularizer
        )(x)
        skip.append(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=(3, 3), activation='relu', kernel_regularizer=None):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = keras.layers.Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same',
                       dilation_rate=2**i,
                       kernel_regularizer=kernel_regularizer)(x)
            dilated_layers.append(x)
        return keras.layers.Add()(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                keras.layers.Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same',
                       dilation_rate=2**i,
                       kernel_regularizer=kernel_regularizer)(x)
            )
        return keras.layers.Add()(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu', kernel_regularizer=None):
    for i in reversed(range(n_block)):
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)

        sk = keras.layers.Conv2D(
            K.int_shape(skip[i])[-1] // 2,  # take two times less filters
            (1, 1),
            activation='relu',
            kernel_regularizer=kernel_regularizer
        )(skip[i])  # apply additional conv to skip connection

        x = keras.layers.Concatenate()([sk, x])
        x = keras.layers.Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)
        x = keras.layers.Conv2D(filters * 2**i, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)
    return x


def decoder_v2(x, skip, filters, n_block=3, kernel_size=(3, 3), activation='relu', kernel_regularizer=None):
    for i in reversed(range(n_block)):
        x = keras.layers.UpSampling2D(size=(2, 2))(x)
        x = keras.layers.Conv2D(filters, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)

        sk = keras.layers.Conv2D(
            K.int_shape(skip[i])[-1] // 2,  # take two times less filters
            (1, 1),
            activation='relu',
            kernel_regularizer=kernel_regularizer
        )(skip[i])  # apply additional conv to skip connection

        x = keras.layers.Concatenate()([sk, x])
        x = keras.layers.Conv2D(filters, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)
        x = keras.layers.Conv2D(filters, kernel_size, activation=activation, padding='same',
                  kernel_regularizer=kernel_regularizer)(x)
    return x


def get_dilated_unet(
        inputs,
        mode='cascade',
        filters=64,
        n_block=3,
        n_class=89,
        bottleneck_depth=2,
        last_name=None,
        decoder_filters=None,
        filters_bottleneck=None,
        kernel_regularizer=None
    ):
    # encoder
    enc, skip = encoder(inputs, filters, n_block, kernel_regularizer=kernel_regularizer)
    
    # bottleneck
    if filters_bottleneck is None:
        filters_bottleneck = filters * 2**n_block
    bottle = bottleneck(
        enc, filters_bottleneck=filters_bottleneck, mode=mode, depth=bottleneck_depth, kernel_regularizer=kernel_regularizer)
    
    # decoder
    if decoder_filters is None:
        dec = decoder(bottle, skip, filters, n_block, kernel_regularizer=kernel_regularizer)
    else:
        dec = decoder_v2(bottle, skip, decoder_filters, n_block, kernel_regularizer=kernel_regularizer)
    
    # last conv
    classify = keras.layers.Conv2D(
        n_class, (1, 1), activation=None, name=last_name, kernel_regularizer=kernel_regularizer)(dec)

    return classify


def get_FBP_models(
        input_shape=(224, 224, 3),
        kernel_regularizer=None,
        weights='imagenet',
        densepose_parts=15,
        densepose_patches=24,
        num_joints=17,
        base_last_feats=384,
        with_loss=True
    ):

    # base network with base imagenet pretrained features
    mdl, layers_to_freeze = create_poor_mans_fpn(
        input_shape=input_shape,
        alpha=1.4,
        kernel_regularizer=kernel_regularizer,
        last_feats=base_last_feats
    )
    F = mdl.output

    # unet for binary segmentation
    B = get_dilated_unet(
        F,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=16,#1,
        bottleneck_depth=3,
        filters_bottleneck=None,
        kernel_regularizer=kernel_regularizer,
        last_name='B_2_reduced'#'output_binary_mask'
    )
    B = keras.layers.ReLU(name='B_2_reduced_act')(B)
    
    FB = keras.layers.Concatenate(name='concatenate_FB')([F, B])
    # parts network reuses base features and binary seg result
    P = get_dilated_unet(
        FB,
        mode='cascade',
        filters=32,
        n_block=3,
        n_class=24,#densepose_parts,
        bottleneck_depth=4,
        filters_bottleneck=192,
        kernel_regularizer=kernel_regularizer,
        last_name='P_2_reduced'#'output_densepose_mask'
    )
    P = keras.layers.ReLU(name='P_2_reduced_act')(P)
    
    # actual high res outputs
    B_out = keras.layers.UpSampling2D(name='B_up')(B)  # upsample
    B_out = conv_block(  # conv to smooth things out
        B_out, 16, kernel_size=(5, 5),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    B_out = keras.layers.Conv2D(
        1,
        (1, 1),
        activation=None,
        name='output_binary_mask',
        kernel_regularizer=kernel_regularizer
    )(B_out)
    
    P_out = keras.layers.UpSampling2D()(P)
    P_out = conv_block(  # conv to smooth things out
        P_out, 24, kernel_size=(5, 5),
        kernel_regularizer=kernel_regularizer,
        padding='same', activation='relu'
    )
    P_out = keras.layers.Conv2D(
        densepose_parts,
        (1, 1),
        activation=None,
        name='output_densepose_mask',
        kernel_regularizer=kernel_regularizer
    )(P_out)

    # GT inputs ##########################################################
    input_binary_mask = keras.layers.Input(shape=(input_shape[0], input_shape[1], 1), name='bin_mask')
    input_densepose_mask = keras.layers.Input(shape=(input_shape[0], input_shape[1], densepose_parts), name='dp_mask')
    # custom loss ########
    if with_loss:
        loss = compute_losses_FBP(
            [B_out, P_out],
            [input_binary_mask, input_densepose_mask]
        )

    joint_model = keras.models.Model([mdl.input, input_binary_mask, input_densepose_mask], [B_out, P_out])
    if with_loss:
        joint_model.add_loss(loss)
    return joint_model, layers_to_freeze


def compute_losses_FBP(preds, truths):
    bin_mask_true, dp_mask_true = truths
    bin_mask_pred, dp_mask_pred = preds
    w1, w2 = 1., 1.

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

    return (
        w1 * bin_mask_loss +
        w2 * dp_mask_loss
    )
