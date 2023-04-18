#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Conv3D, Input, MaxPooling2D, Flatten, add, Activation, ConvLSTM2D, \
    TimeDistributed, Bidirectional, Reshape, InputLayer, LSTM
from keras.layers.normalization import BatchNormalization
from keras.metrics import binary_accuracy
from keras.models import Sequential

import warnings

warnings.filterwarnings("ignore")

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

"""
def resBlock(ipt, filters, increDimen=False):

    #Residual blocks for extracting more deep, effective and distinguishable features.

    res = ipt

    if increDimen:
        ipt = MaxPooling2D(pool_size=(2, 2), padding="same")(ipt)
        res = Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)

    out = BatchNormalization()(ipt)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = add([res, out])

    return out
"""


# 通道注意力机制
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    shared_layer_one = Dense(channel // ratio,
                             kernel_initializer='he_normal',
                             activation='relu',
                             use_bias=True,
                             bias_initializer='zeros')

    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)  # 将输入的维度按照给定模式进行重排

    return multiply([input_feature, cbam_feature])  # cbam_feature输入乘以input_feature输入的每一层


# 空间注意力机制
def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])  # 拼接
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in CBAM: Convolutional Block Attention Module.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature, )
    return cbam_feature


def MGF6mARice():
    """

    """
    num_nodes = 11  # # maxNumAtoms of G == 11
    num_features = 17  # # according to the number of features of each atom, which features used can be self-define
    seqLength = 16  #
    dropout1 = 0.285293526161375
    dropout2 = 0.746951714170157

    # Input features about molecular graph features
    features = Input(shape=(seqLength, num_nodes, num_features))

    # Convolutional layer at the beginning for preliminary feature extraction
    conv2dLayer = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", name='conv2d')(features)
    conv2dLayer = BatchNormalization()(conv2dLayer)
    cbam = cbam_block(conv2dLayer)
    x = add([conv2dLayer, cbam])

  


    flattenLayer = Flatten()(x)

    # MLP to build a prediction
    dense1Layer = Dense(256, activation='relu', name='dense1')(flattenLayer)
    dropout1Layer = Dropout(rate=dropout1, name='dropout1')(dense1Layer)
    dense2Layer = Dense(64, kernel_initializer='glorot_normal', activation='relu', name='dense2')(dropout1Layer)
    dropout2Layer = Dropout(rate=dropout2, name='dropout2')(dense2Layer)
    dense3layer = Dense(32, activation='relu', name='dense3')(dropout2Layer)


    pred = Dense(1, activation='sigmoid', name='dense4')(dense3layer)

    model = Model(inputs=features, outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[binary_accuracy])

    print(model.summary())

    return model