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

from activity_data_provider_layer import ActivityDataProvider
from build_val_model import act_proto, exp_proto
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

def pj_x(mode, batchsize, exp_T, exp_vocab_size):
    n = caffe.NetSpec()
    mode_str = json.dumps({'mode':mode, 'batchsize':batchsize})
    n.img_feature, n.label, n.exp, n.exp_out, n.exp_cont_1, n.exp_cont_2 = \
        L.Python(module='activity_data_provider_layer',
                 layer='ActivityDataProviderLayer',
                 param_str=mode_str, ntop=6)


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
    n.prediction = L.InnerProduct(n.att_feature_resh, num_output=config.NUM_OUTPUT_UNITS, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.prediction, n.label)
    n.accuracy = L.Accuracy(n.prediction, n.label)

    # Embed Activity GT answer during training
    n.exp_emb_ans = L.Embed(n.label, input_dim=config.NUM_OUTPUT_UNITS, num_output=300, \
        weight_filler=dict(type='uniform', min=-0.08, max=0.08))
    n.exp_emb_ans_tanh = L.TanH(n.exp_emb_ans)
    n.exp_emb_ans2 = L.InnerProduct(n.exp_emb_ans_tanh, num_output=2048, weight_filler=dict(type='xavier'))

    # merge activity answer and visual feature
    n.exp_emb_resh = L.Reshape(n.exp_emb_ans2, reshape_param=dict(shape=dict(dim=[-1,2048,1,1])))
    n.exp_emb_tiled_1 = L.Tile(n.exp_emb_resh, axis=2, tiles=14)
    n.exp_emb_tiled = L.Tile(n.exp_emb_tiled_1, axis=3, tiles=14)

    n.img_embed = L.Convolution(n.img_feature, kernel_size=1, stride=1, num_output=2048, pad=0, weight_filler=dict(type='xavier'))
    n.exp_eltwise = L.Eltwise(n.img_embed,  n.exp_emb_tiled, eltwise_param={'operation': P.Eltwise.PROD})
    n.exp_eltwise_sqrt = L.SignedSqrt(n.exp_eltwise)
    n.exp_eltwise_l2 = L.L2Normalize(n.exp_eltwise_sqrt)
    n.exp_eltwise_drop = L.Dropout(n.exp_eltwise_l2, dropout_param={'dropout_ratio': 0.3})

    # Attention for Explanation
    n.exp_att_conv1 = L.Convolution(n.exp_eltwise_drop, kernel_size=1,
                                    stride=1, num_output=512, pad=0, weight_filler=dict(type='xavier'))
    n.exp_att_conv1_relu = L.ReLU(n.exp_att_conv1)
    n.exp_att_conv2 = L.Convolution(n.exp_att_conv1_relu, kernel_size=1,
                                    stride=1, num_output=1, pad=0, weight_filler=dict(type='xavier'))
    n.exp_att_reshaped = L.Reshape(n.exp_att_conv2,reshape_param=dict(shape=dict(dim=[-1,1,14*14])))
    n.exp_att_softmax = L.Softmax(n.exp_att_reshaped, axis=2)
    n.exp_att_map = L.Reshape(n.exp_att_softmax,reshape_param=dict(shape=dict(dim=[-1,1,14,14])))

    exp_dummy = L.DummyData(shape=dict(dim=[batchsize, 1]), data_filler=dict(type='constant', value=1), ntop=1)
    n.exp_att_feature_prev = L.SoftAttention(n.img_feature, n.exp_att_map, exp_dummy)
    n.exp_att_feature_resh = L.Reshape(n.exp_att_feature_prev, reshape_param=dict(shape=dict(dim=[-1, 2048])))
    n.exp_att_feature_embed = L.InnerProduct(n.exp_att_feature_resh, num_output=2048, weight_filler=dict(type='xavier'))
    n.exp_att_feature = L.Eltwise(n.exp_emb_ans2, n.exp_att_feature_embed, eltwise_param={'operation': P.Eltwise.PROD})

    # Embed explanation
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

    n.exp_loss = L.SoftmaxWithLoss(n.exp_prediction, n.exp_out,
                                   loss_param=dict(ignore_label=-1),
                                   softmax_param=dict(axis=2))
    n.exp_accuracy = L.Accuracy(n.exp_prediction, n.exp_out, axis=2, ignore_label=-1)

    return n.to_proto()

def make_answer_vocab(adic, vocab_size):
    """
    Returns a dictionary that maps words to indices.
    """
    adict = {}
    id = 0
    for qid in adic.keys():
        answer = adic[qid]
        if answer in adict:
            continue
        else:
            adict[answer] = id 
            id +=1
    
    return adict


def make_exp_vocab(exp_dic):
    """
    Returns a dictionary that maps words to indices.
    """
    exp_vdict = {'<EOS>': 0}
    exp_vdict[''] = 1
    exp_id = 2
    for qid in exp_dic.keys():
        exp_strings = exp_dic[qid]
        for exp_str in exp_strings:
            exp_list = ActivityDataProvider.seq_to_list(exp_str)

            for w in exp_list:
                if w not in exp_vdict:
                    exp_vdict[w] = exp_id
                    exp_id += 1

    return exp_vdict


def make_vocab_files():
    """
    Produce the answer and explanation vocabulary files.
    """
    print('making answer vocab...', config.ANSWER_VOCAB_SPACE)
    _, adic, _ = ActivityDataProvider.load_data(config.ANSWER_VOCAB_SPACE)
    answer_vocab = make_answer_vocab(adic, config.NUM_OUTPUT_UNITS)
    print('making explanation vocab...', config.EXP_VOCAB_SPACE)
    _, _, expdic = ActivityDataProvider.load_data(config.EXP_VOCAB_SPACE)
    explanation_vocab = make_exp_vocab(expdic)
    return answer_vocab, explanation_vocab

def reverse(dict):
    rev_dict = {}
    for k, v in dict.items():
        rev_dict[v] = k
    return rev_dict

def to_str(type, idxs, cont, r_adict, r_exp_vdict):
    if type == 'a':
        return r_adict[idxs]
    elif type == 'exp':
        words = []
        for idx in idxs:
            if idx == 0:
                break
            words.append(r_exp_vdict[idx])

    return ' '.join(words)

def batch_to_str(type, batch_idx, batch_cont, r_adict, r_exp_vdict):

    converted = []
    for idxs, cont in zip(batch_idx, batch_cont):
        converted.append(to_str(type, idxs, cont, r_adict, r_exp_vdict))
    return converted
        
def main():
    if not os.path.exists('./model'):
        os.makedirs('./model')

    answer_vocab, explanation_vocab = {}, {}
    if os.path.exists('./model/adict.json') and os.path.exists('./model/exp_vdict.json'):
        print('restoring vocab')
        with open('./model/adict.json','r') as f:
            answer_vocab = json.load(f)
        with open('./model/exp_vdict.json','r') as f:
            exp_vocab = json.load(f)
    else:
        answer_vocab, exp_vocab = make_vocab_files()
        with open('./model/adict.json','w') as f:
            json.dump(answer_vocab, f)
        with open('./model/exp_vdict.json','w') as f:
            json.dump(exp_vocab, f)

    r_adict = reverse(answer_vocab)
    r_exp_vdict = reverse(exp_vocab)


    print('answer vocab size:', len(answer_vocab))
    print('exp vocab size:', len(exp_vocab))

    with open('./model/proto_train.prototxt', 'w') as f:
        f.write(str(pj_x(config.TRAIN_DATA_SPLITS, config.BATCH_SIZE, \
            config.MAX_WORDS_IN_EXP, len(exp_vocab))))
    
    with open('./model/act_proto_test_gt.prototxt', 'w') as f:
        f.write(str(act_proto('val', config.VAL_BATCH_SIZE, len(exp_vocab), use_gt=True)))

    with open('./model/act_proto_test_pred.prototxt', 'w') as f:
        f.write(str(act_proto('val', config.VAL_BATCH_SIZE, len(exp_vocab), use_gt=False)))

    with open('./model/exp_proto_test.prototxt', 'w') as f:
        f.write(str(exp_proto('val', config.VAL_BATCH_SIZE, 1, len(exp_vocab))))

    caffe.set_device(config.GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.get_solver('./pj_x_solver.prototxt')

    train_loss = np.zeros(config.MAX_ITERATIONS)
    train_acc = np.zeros(config.MAX_ITERATIONS)
    train_loss_exp = np.zeros(config.MAX_ITERATIONS)
    train_acc_exp = np.zeros(config.MAX_ITERATIONS)
    results = []

    for it in range(config.MAX_ITERATIONS):
        solver.step(1)
    
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        train_acc[it] = solver.net.blobs['accuracy'].data
        train_loss_exp[it] = solver.net.blobs['exp_loss'].data
        train_acc_exp[it] = solver.net.blobs['exp_accuracy'].data
   
        if it != 0 and it % config.PRINT_INTERVAL == 0:
            print('Iteration:', it)
            c_mean_loss = train_loss[it-config.PRINT_INTERVAL:it].mean()
            c_mean_acc = train_acc[it-config.PRINT_INTERVAL:it].mean()
            c_mean_loss_exp = train_loss_exp[it-config.PRINT_INTERVAL:it].mean()
            c_mean_acc_exp = train_acc_exp[it-config.PRINT_INTERVAL:it].mean()
            print('Train loss for classification:', c_mean_loss)
            print('Train accuracy for classification:', c_mean_acc)
            print('Train loss for exp:', c_mean_loss_exp)
            print('Train accuracy for exp:', c_mean_acc_exp)

            predicted_ans = solver.net.blobs['prediction'].data
            predicted_ans = predicted_ans.argmax(axis=1)
            answers = solver.net.blobs['label'].data
            generated_exp = solver.net.blobs['exp_prediction'].data
            generated_exp = generated_exp.argmax(axis=2).transpose()
            target_exp = solver.net.blobs['exp_out'].data.transpose()
            exp_out_cont = solver.net.blobs['exp_cont_2'].data.transpose()

            predict_str = batch_to_str('a', predicted_ans, np.ones_like(predicted_ans),
                                       r_adict, r_exp_vdict)
            answers_str = batch_to_str('a', answers, np.ones_like(answers),
                                       r_adict, r_exp_vdict)
            generated_str = batch_to_str('exp', generated_exp, exp_out_cont,
                                         r_adict, r_exp_vdict)
            target_str = batch_to_str('exp', target_exp, exp_out_cont,
                                      r_adict, r_exp_vdict)

            count = 0
            for pred, ans, exp, target in zip(predict_str, answers_str, generated_str, target_str):
                if count == 10:
                    break
                print('Pred:', pred)
                print('A:', ans)
                print('Because...')
                print('\tgenerated:', exp)
                print('\ttarget:', target)
                count += 1

if __name__ == '__main__':
    main()
