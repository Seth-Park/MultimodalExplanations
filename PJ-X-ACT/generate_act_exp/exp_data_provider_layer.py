import caffe
import numpy as np
import re, json, random

CURRENT_DATA_SHAPE = None
MAX_WORDS_IN_EXP = None

class ExpDataProviderLayer(caffe.Layer):
    """
    Provide input data for Explanation.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        self.top_names = ['exp_att_feature', 'exp', 'exp_out', 'exp_cont_1', 'exp_cont_2']
        max_exp_words = MAX_WORDS_IN_EXP
        top[0].reshape(self.batchsize, CURRENT_DATA_SHAPE)
        top[1].reshape(max_exp_words, self.batchsize)
        top[2].reshape(max_exp_words, self.batchsize)
        top[3].reshape(max_exp_words, self.batchsize)
        top[4].reshape(max_exp_words, self.batchsize)

        self.mode = json.loads(self.param_str)['mode']

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

