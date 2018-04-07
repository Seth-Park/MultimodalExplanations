import matplotlib
matplotlib.use('Agg')
import os
import sys
import numpy as np
import json
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L
from caffe import params as P

import config


def learning_params(param_list):
    param_dicts = []
    for pl in param_list:
        param_dict = {}
        param_dict['lr_mult'] = pl[0]
        if len(pl) > 1:
            param_dict['decay_mult'] = pl[0]
        param_dicts.append(param_dict)
    return param_dicts

fixed_weights = learning_params([[0, 0], [0, 0]])
fixed_weights_lstm = learning_params([[0, 0], [0, 0], [0, 0]])

def act_proto(mode, batchsize, exp_vocab_size, use_gt=True):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.img_feature, n.label, n.exp, n.exp_out, n.exp_cont_1, n.exp_cont_2 = \
        L.Python(module='activity_data_provider_layer', layer='ActivityDataProviderLayer', param_str=mode_str, ntop=6)

    # Attention
    n.att_conv1 = L.Convolution(n.img_feature, kernel_size=1, stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.att_conv1_relu = L.ReLU(n.att_conv1)
    n.att_conv2 = L.Convolution(n.att_conv1_relu, kernel_size=1, stride=1, num_output=1, pad=0, weight_filler=dict(type='xavier'))
    n.att_reshaped = L.Reshape(n.att_conv2,reshape_param=dict(shape=dict(dim=[-1,1,14*14])))
    n.att_softmax = L.Softmax(n.att_reshaped, axis=2)
    n.att_map = L.Reshape(n.att_softmax,reshape_param=dict(shape=dict(dim=[-1,1,14,14])))
    
    dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.att_feature  = L.SoftAttention(n.img_feature, n.att_map, dummy)
    n.att_feature_resh = L.Reshape(n.att_feature, reshape_param=dict(shape=dict(dim=[-1,2048])))

    # Prediction
    n.prediction = L.InnerProduct(n.att_feature_resh, num_output=config.NUM_OUTPUT_UNITS, weight_filler=dict(type='xavier'), param=fixed_weights)

    # Take GT answer or Take the logits of the VQA model and get predicted answer to embed
    if use_gt:
        n.exp_emb_ans = L.Embed(n.label, input_dim=config.NUM_OUTPUT_UNITS, num_output=300,
            weight_filler=dict(type='uniform', min=-0.08, max=0.08))
    else:
        n.vqa_ans = L.ArgMax(n.prediction, axis=1)
        n.exp_emb_ans = L.Embed(n.vqa_ans, input_dim=config.NUM_OUTPUT_UNITS, num_output=300,
            weight_filler=dict(type='uniform', min=-0.08, max=0.08))
    n.exp_emb_ans_tanh = L.TanH(n.exp_emb_ans)
    n.exp_emb_ans2 = L.InnerProduct(n.exp_emb_ans_tanh, num_output=2048, weight_filler=dict(type='xavier'))

    # Merge activity answer and visual feature
    n.exp_emb_resh = L.Reshape(n.exp_emb_ans2, reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.exp_emb_tiled_1 = L.Tile(n.exp_emb_resh, axis=2, tiles=14)
    n.exp_emb_tiled = L.Tile(n.exp_emb_tiled_1, axis=3, tiles=14)

    n.img_embed = L.Convolution(n.img_feature, kernel_size=1, stride=1, num_output=2048, pad=0, weight_filler=dict(type='xavier'))
    n.exp_eltwise = L.Eltwise(n.img_embed,  n.exp_emb_tiled, eltwise_param={'operation': P.Eltwise.PROD})
    n.exp_eltwise_sqrt = L.SignedSqrt(n.exp_eltwise)
    n.exp_eltwise_l2 = L.L2Normalize(n.exp_eltwise_sqrt)
    n.exp_eltwise_drop = L.Dropout(n.exp_eltwise_l2, dropout_param={'dropout_ratio': 0.3})

    # Attention for Explanation
    n.exp_att_conv1 = L.Convolution(n.exp_eltwise_drop, kernel_size=1, stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.exp_att_conv1_relu = L.ReLU(n.exp_att_conv1)
    n.exp_att_conv2 = L.Convolution(n.exp_att_conv1_relu, kernel_size=1, stride=1, num_output=1, pad=0, weight_filler=dict(type='xavier'))
    n.exp_att_reshaped = L.Reshape(n.exp_att_conv2,reshape_param=dict(shape=dict(dim=[-1,1,14*14])))
    n.exp_att_softmax = L.Softmax(n.exp_att_reshaped, axis=2)
    n.exp_att_map = L.Reshape(n.exp_att_softmax,reshape_param=dict(shape=dict(dim=[-1,1,14,14])))
    
    exp_dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.exp_att_feature_prev  = L.SoftAttention(n.img_feature, n.exp_att_map, exp_dummy)
    n.exp_att_feature_resh = L.Reshape(n.exp_att_feature_prev, reshape_param=dict(shape=dict(dim=[-1, 2048])))
    n.exp_att_feature_embed = L.InnerProduct(n.exp_att_feature_resh, num_output=2048, weight_filler=dict(type='xavier'))
    n.exp_att_feature = L.Eltwise(n.exp_emb_ans2, n.exp_att_feature_embed, eltwise_param={'operation': P.Eltwise.PROD})

    n.silence_exp_att = L.Silence(n.exp_att_feature, ntop=0)

    return n.to_proto()

def exp_proto(mode, batchsize, exp_T, exp_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.exp_att_feature, n.exp, n.exp_out, n.exp_cont_1, n.exp_cont_2 = \
        L.Python(module='exp_data_provider_layer', layer='ExpDataProviderLayer', param_str=mode_str, ntop=5)

    n.exp_embed_ba = L.Embed(n.exp, input_dim=exp_vocab_size, num_output=300, \
        weight_filler=dict(type='uniform', min=-0.08, max=0.08))
    n.exp_embed = L.TanH(n.exp_embed_ba)


    # LSTM1 for Explanation
    n.exp_lstm1 = L.LSTM(\
                   n.exp_embed, n.exp_cont_1,\
                   recurrent_param=dict(\
                       num_output=2048,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))

    n.exp_lstm1_dropped = L.Dropout(n.exp_lstm1,dropout_param={'dropout_ratio':0.3})

    # merge with LSTM1 for explanation
    n.exp_att_resh = L.Reshape(n.exp_att_feature, reshape_param=dict(shape=dict(dim=[1, -1, 2048])))
    n.exp_att_tiled = L.Tile(n.exp_att_resh, axis=0, tiles=exp_T)
    n.exp_eltwise_all = L.Eltwise(n.exp_lstm1_dropped, n.exp_att_tiled, eltwise_param={'operation': P.Eltwise.PROD})
    n.exp_eltwise_all_l2 = L.L2Normalize(n.exp_eltwise_all)
    n.exp_eltwise_all_drop = L.Dropout(n.exp_eltwise_all_l2, dropout_param={'dropout_ratio': 0.3})

    # LSTM2 for Explanation
    n.exp_lstm2 = L.LSTM(\
                   n.exp_eltwise_all_drop, n.exp_cont_2,\
                   recurrent_param=dict(\
                       num_output=1024,\
                       weight_filler=dict(type='uniform',min=-0.08,max=0.08),\
                       bias_filler=dict(type='constant',value=0)))
    n.exp_lstm2_dropped = L.Dropout(n.exp_lstm2,dropout_param={'dropout_ratio':0.3})
    
    n.exp_prediction = L.InnerProduct(n.exp_lstm2_dropped, num_output=exp_vocab_size, weight_filler=dict(type='xavier'), axis=2)

    n.silence_exp_prediction = L.Silence(n.exp_prediction, ntop=0)


    return n.to_proto()
