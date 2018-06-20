import matplotlib
matplotlib.use('Agg')
import os
import sys
import random
import argparse
import numpy as np
import caffe
import glob
import json

import activity_data_provider_layer
import exp_data_provider_layer

from collections import OrderedDict
from activity_data_provider_layer import ActivityDataProvider
from util import *

# UNK is the word used to identify unknown words
UNK = '<unk>'

def verify_folder(folder_path, use_gt=True):
    """
    Makes sure all the required files exist in the folder. If so, returns the
    paths to all the files.
    """
    if use_gt:
        act_proto_path = folder_path + '/act_proto_test_gt.prototxt'
    else:
        act_proto_path = folder_path + '/act_proto_test_pred.prototxt'

    exp_proto_path = folder_path + '/exp_proto_test.prototxt'
    adict_path = folder_path + '/adict.json'
    exp_vdict_path = folder_path + '/exp_vdict.json'
    assert os.path.exists(act_proto_path), 'act_proto_test.prototxt missing'
    assert os.path.exists(exp_proto_path), 'exp_proto_test.prototxt missing'
    assert os.path.exists(adict_path), 'adict.json missing'
    assert os.path.exists(exp_vdict_path), 'exp_vdict.json missing'

    return act_proto_path, exp_proto_path, adict_path, exp_vdict_path

def generate_sentences(args):
    act_proto_path, exp_proto_path, adict_path, exp_vdict_path = \
        verify_folder(args.folder, args.use_gt)

    dp = ActivityDataProvider(args.ann_file,
                              adict_path, exp_vdict_path,
                              args.batch_size, args.data_shape, args.img_feature_prefix,
                              args.exp_max_length, mode='val')
    total_questions = len(dp.getQuesIds())
    print(total_questions, 'total questions')

    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    activity_data_provider_layer.CURRENT_DATA_SHAPE = args.data_shape
    activity_data_provider_layer.MAX_WORDS_IN_EXP = args.exp_max_length

    exp_data_provider_layer.CURRENT_DATA_SHAPE = args.data_shape[0]
    exp_data_provider_layer.MAX_WORDS_IN_EXP = 1  # predict one by one

    act_net = caffe.Net(act_proto_path, args.model_path, caffe.TEST)
    exp_net = caffe.Net(exp_proto_path, args.model_path, caffe.TEST)
    print('ACT model loaded:', act_proto_path, args.model_path)
    print('EXP model loaded:', exp_proto_path, args.model_path)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.save_att_map:
        att_map_save_dir = os.path.join(args.out_dir, 'att_maps')
        if not os.path.exists(att_map_save_dir):
            os.makedirs(att_map_save_dir)

    with open(args.ann_file, 'r') as f:
        exp_anno = json.load(f)

    final_results = {}
    epoch = 0
    predictions = []
    ground_truths = []
    while epoch == 0:
        ivec, avec, exp_vec, exp_vec_out, exp_cvec_1, exp_cvec_2, \
        qid_list, iid_list, epoch = dp.get_batch_vec()
        shape = ivec.shape
        if act_net.blobs['img_feature'].data.shape != shape:
            act_net.blobs['img_feature'].reshape(*shape)
            act_net.blobs['DummyData1'].reshape(shape[0], 1)
            act_net.blobs['DummyData2'].reshape(shape[0], 1)
            act_net.blobs['label'].reshape(*avec.shape)
            act_net.blobs['exp'].reshape(*exp_vec.transpose().shape)
            act_net.blobs['exp_out'].reshape(*exp_vec_out.transpose().shape)
            act_net.blobs['exp_cont_1'].reshape(*exp_cvec_1.transpose().shape)
            act_net.blobs['exp_cont_2'].reshape(*exp_cvec_2.transpose().shape)
        act_net.blobs['img_feature'].data[...] = ivec
        act_net.blobs['label'].data[...] = avec
        act_net.blobs['exp'].data[...] = exp_vec.transpose()           # not used
        act_net.blobs['exp_out'].data[...] = exp_vec_out.transpose()   # not used
        act_net.blobs['exp_cont_1'].data[...] = exp_cvec_1.transpose() # not used
        act_net.blobs['exp_cont_2'].data[...] = exp_cvec_2.transpose() # not used

        act_net.forward()

        act_predictions = act_net.blobs['prediction'].data.copy()
        act_preds = act_predictions.argmax(axis=1)
        exp_att_feature = act_net.blobs['exp_att_feature'].data.copy()
        exp_att_feature = np.squeeze(exp_att_feature)

        predictions.append(act_preds.copy())
        ground_truths.append(avec.copy())

        act_att_map = act_net.blobs['att_map'].data.copy()
        exp_att_map = act_net.blobs['exp_att_map'].data.copy()
        if args.save_att_map:
            save_att_map(iid_list, exp_att_map, att_map_save_dir)

        finished = np.zeros(args.batch_size)
        predicted_words = []
        conts = []
        t = 0
        prev_word = exp_vec[:, 0].reshape((1, args.batch_size)) # Initialize with <SOS>
        continuation = np.zeros((1, args.batch_size)) # flush out for the first word

        while finished.sum() != args.batch_size and t < args.exp_max_length:
            shape = exp_att_feature.shape
            if exp_net.blobs['exp_att_feature'].data.shape != shape:
                exp_net.blobs['exp_att_feature'].reshape(*shape)
                exp_net.blobs['exp'].reshape(*prev_word.shape)
                exp_net.blobs['exp_out'].reshape(1, args.batch_size)
                exp_net.blobs['exp_cont_1'].reshape(1, args.batch_size)
                exp_net.blobs['exp_cont_2'].reshape(1, args.batch_size)
            exp_net.blobs['exp_att_feature'].data[...] = exp_att_feature
            exp_net.blobs['exp'].data[...] = prev_word
            exp_net.blobs['exp_out'].data[...] = exp_vec_out[:, t].reshape((1, args.batch_size))
            exp_net.blobs['exp_cont_1'].data[...] = continuation
            exp_net.blobs['exp_cont_2'].data[...] = continuation
            exp_net.forward()
            predicted_word = exp_net.blobs['exp_prediction'].data.copy()
            predicted_word = np.squeeze(predicted_word.argmax(axis=2))

            completed = np.where(predicted_word == 0)
            finished[completed] = 1
            predicted_words.append(predicted_word)
            conts.append(continuation)
            prev_word = predicted_word.reshape((1, args.batch_size))
            continuation = (finished != 1).astype(np.int32).reshape((1, args.batch_size))
            t += 1

        predicted_words = np.array(predicted_words).transpose()
        conts = np.array(conts).transpose()

        r_adict = reverse(dp.adict)
        r_exp_vdict = reverse(dp.exp_vdict)

        predict_str = batch_to_str('a', act_preds, np.ones_like(act_preds), r_adict, r_exp_vdict)
        answers_str = batch_to_str('a', avec, np.ones_like(avec), r_adict, r_exp_vdict)
        generated_str = batch_to_str('exp', predicted_words, conts, r_adict, r_exp_vdict)

        for qid, pred, ans, expl, act_att, exp_att in zip(qid_list, predict_str,
                                                          answers_str, generated_str,
                                                          act_att_map, exp_att_map):
            if ans == '':
                ans = UNK
            iid = dp.getImgId(qid)
            true_explanations = exp_anno[qid]['exp']
            final_results[qid] = {'ans': ans, 'exp': expl, 'pred': pred}

    with open(os.path.join(args.out_dir, 'exp_results.json'), 'w') as f:
        json.dump(final_results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', default=None, help='activity annotation file')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--folder', required=True, help='folder containing model')
    parser.add_argument('--model_path', required=True, help='path to caffemodel')
    parser.add_argument('--use_gt', action='store_true',
        help='whether to use ground-truth answer when generating explanations')
    parser.add_argument('--save_att_map', action='store_true',
        help='whether to store attention maps in numpy format')
    parser.add_argument('--batch_size', default=100, type=int, help='test batch size')
    parser.add_argument('--max_length', default=15,
        type=int, help='max time step for question encoding LSTM')
    parser.add_argument('--exp_max_length', default=36,
        type=int, help='max time step for explanation generating LSTM')
    parser.add_argument('--data_shape', default=(2048, 14, 14),
        help='shape of the input image feature')
    parser.add_argument('--img_feature_prefix', default='../ACT-X/Features/resnet_res5c_bgrms_large/')
    args = parser.parse_args()
    assert len(args.folder) > 0, 'please specify one folder'

    generate_sentences(args)

if __name__ == '__main__':
    main()

    
