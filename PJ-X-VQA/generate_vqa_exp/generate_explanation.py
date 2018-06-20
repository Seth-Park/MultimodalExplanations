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

import vqa_data_provider_layer
import exp_data_provider_layer

from collections import OrderedDict
from vqa_data_provider_layer import VQADataProvider
from util import *

# UNK is the word used to identify unknown words
UNK = '<unk>'

def verify_folder(folder_path, use_gt=True):
    """
    Makes sure all the required files exist in the folder. If so, returns the
    paths to all the files.
    """
    if use_gt:
        vqa_proto_path = folder_path + '/vqa_proto_test_gt.prototxt'
    else:
        vqa_proto_path = folder_path + '/vqa_proto_test_pred.prototxt'

    exp_proto_path = folder_path + '/exp_proto_test.prototxt'
    adict_path = folder_path + '/adict.json'
    vdict_path = folder_path + '/vdict.json'
    exp_vdict_path = folder_path + '/exp_vdict.json'
    assert os.path.exists(vqa_proto_path), 'vqa_proto_test.prototxt missing'
    assert os.path.exists(exp_proto_path), 'exp_proto_test.prototxt missing'
    assert os.path.exists(adict_path), 'adict.json missing'
    assert os.path.exists(vdict_path), 'vdict.json missing'
    assert os.path.exists(exp_vdict_path), 'exp_vdict.json missing'

    return vqa_proto_path, exp_proto_path, adict_path, vdict_path, exp_vdict_path

def generate_sentences(args):
    vqa_proto_path, exp_proto_path, adict_path, vdict_path, exp_vdict_path = \
        verify_folder(args.folder, args.use_gt)
    model_path = args.model_path

    dp = VQADataProvider(args.ques_file, args.ann_file, args.exp_file,
                         vdict_path, adict_path, exp_vdict_path,
                         args.batch_size, args.data_shape, args.img_feature_prefix,
                         args.max_length, args.exp_max_length, mode='val')
    total_questions = len(dp.getQuesIds())
    print(total_questions, 'total questions')

    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()

    vqa_data_provider_layer.CURRENT_DATA_SHAPE = args.data_shape
    vqa_data_provider_layer.MAX_WORDS_IN_QUESTION = args.max_length
    vqa_data_provider_layer.MAX_WORDS_IN_EXP = args.exp_max_length

    exp_data_provider_layer.CURRENT_DATA_SHAPE = args.data_shape[0]
    exp_data_provider_layer.MAX_WORDS_IN_EXP = 1  # predict one by one

    vqa_net = caffe.Net(vqa_proto_path, args.model_path, caffe.TEST)
    exp_net = caffe.Net(exp_proto_path, args.model_path, caffe.TEST)
    print('VQA model loaded:', vqa_proto_path, args.model_path)
    print('EXP model loaded:', exp_proto_path, args.model_path)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.save_att_map:
        att_map_save_dir = os.path.join(args.out_dir, 'att_maps')
        if not os.path.exists(att_map_save_dir):
            os.makedirs(att_map_save_dir)

    with open(args.exp_file, 'r') as f:
        exp_anno = json.load(f)

    final_results = {}
    vqa_submit_results = []
    epoch = 0
    while epoch == 0:
        qvec, cvec, ivec, avec, exp_vec, exp_vec_out, exp_cvec_1, exp_cvec_2, \
        qid_list, _, epoch = dp.get_batch_vec()
        shape = ivec.shape
        if vqa_net.blobs['img_feature'].data.shape != shape:
            vqa_net.blobs['img_feature'].reshape(*shape)
            vqa_net.blobs['data'].reshape(*np.transpose(qvec, (1, 0)).shape)
            vqa_net.blobs['cont'].reshape(*np.transpose(cont, (1, 0)).shape)
            vqa_net.blobs['label'].reshape(*avec.shape)
            vqa_net.blobs['exp'].reshape(exp_vec.transpose().shape)
            vqa_net.blobs['exp_out'].reshape(exp_vec_out.transpose().shape)
            vqa_net.blobs['exp_cont_1'].reshape(exp_cvec_1.transpose().shape)
            vqa_net.blobs['exp_cont_2'].reshape(exp_cvec_2.transpose().shape)

        vqa_net.blobs['data'].data[...] = np.transpose(qvec, (1, 0))
        vqa_net.blobs['cont'].data[...] = np.transpose(cvec, (1, 0))
        vqa_net.blobs['img_feature'].data[...] = ivec
        vqa_net.blobs['label'].data[...] = avec
        vqa_net.blobs['exp'].data[...] = exp_vec.transpose()           # not used
        vqa_net.blobs['exp_out'].data[...] = exp_vec_out.transpose()   # not used
        vqa_net.blobs['exp_cont_1'].data[...] = exp_cvec_1.transpose() # not used
        vqa_net.blobs['exp_cont_2'].data[...] = exp_cvec_2.transpose() # not used

        vqa_net.forward()

        vqa_predictions = vqa_net.blobs['prediction'].data.copy()
        vqa_preds = vqa_predictions.argmax(axis=1)
        exp_att_feature = vqa_net.blobs['exp_att_feature'].data.copy()
        exp_att_feature = np.squeeze(exp_att_feature)

        vqa_att_map = vqa_net.blobs['att_map'].data.copy()
        exp_att_map = vqa_net.blobs['exp_att_map'].data.copy()
        if args.save_att_map:
            save_att_map(qid_list, exp_att_map, att_map_save_dir)

        finished = np.zeros(args.batch_size)
        predicted_words = []
        conts = []
        t = 0
        prev_word = exp_vec[:, 0].reshape((1, args.batch_size)) # Initialize with <SOS>
        continuation = np.zeros((1, args.batch_size)) # flush out for the first word

        while finished.sum() != args.batch_size and t < args.exp_max_length:
            shape = exp_att_feature.shape
            if exp_net.blobs['exp_att_feature'].data.shape != shape:
                exp_net.blob['exp_att_feature'].reshape(*shape)
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

        r_vdict = reverse(dp.vdict)
        r_adict = reverse(dp.adict)
        r_exp_vdict = reverse(dp.exp_vdict)

        questions_str = batch_to_str('q', qvec, cvec, r_vdict, r_adict, r_exp_vdict)
        answers_str = batch_to_str('a', avec, np.ones_like(avec), r_vdict, r_adict, r_exp_vdict)
        pred_str = batch_to_str('a', vqa_preds, np.ones_like(vqa_preds), r_vdict, r_adict, r_exp_vdict)
        generated_str = batch_to_str('exp', predicted_words, conts, r_vdict, r_adict, r_exp_vdict)

        for qid, qstr, ans, pred, expl, vqa_att, exp_att in zip(qid_list, questions_str,
                                                                answers_str, pred_str, generated_str,
                                                                vqa_att_map, exp_att_map):
            if ans == '':
                ans = UNK
            final_results[qid] = {'qstr': qstr, 'ans': ans, 'exp': expl, 'pred': pred}
            vqa_submit_results.append({u'answer': pred, u'question_id': int(qid)})

    with open(os.path.join(args.out_dir, 'exp_results.json'), 'w') as f:
        json.dump(final_results, f)
    with open(os.path.join(args.out_dir, 'vqa_results.json'), 'w') as f:
        json.dump(vqa_submit_results, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ques_file', required=True, help='vqa question file')
    parser.add_argument('--ann_file', default=None, help='vqa annotation file')
    parser.add_argument('--exp_file', required=True, help='exp ques/ann file')
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
    parser.add_argument('--img_feature_prefix', default='../VQA-X/Features/resnet_res5c_bgrms_large/val2014/COCO_val2014_')
    args = parser.parse_args()
    assert len(args.folder) > 0, 'please specify one folder'

    generate_sentences(args)

if __name__ == '__main__':
    main()

    
