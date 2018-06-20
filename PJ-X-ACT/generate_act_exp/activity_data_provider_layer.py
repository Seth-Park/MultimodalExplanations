import caffe
import numpy as np
import re, json, random


CURRENT_DATA_SHAPE = None
MAX_WORDS_IN_EXP = None

class ActivityDataProvider:

    def __init__(self, ann_file,
                 adict_path, exp_vdict_path,
                 batch_size, data_shape, img_feature_prefix, 
                 exp_max_length=36, mode='val'):

        self.ann_file = ann_file
        self.batchsize = batch_size
        self.data_shape = data_shape
        self.img_feature_prefix = img_feature_prefix
        self.exp_max_length = exp_max_length
        self.mode = mode
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None

        # Reading ans file
        print('reading:', self.ann_file)
        with open(self.ann_file, 'r') as f:
            adata = json.load(f)
            qdic = {}
            adic = {}
            exp_dic = {}
            for k, v in adata.items():
                qdic[k] = v['iid']
                adic[k] = v['ans']
                exp_dic[k] = v['exp']
            self.qdic = qdic
            self.adic = adic
            self.expdic = exp_dic
             
        print('parsed', len(self.qdic), 'questions')
        print('parsed', len(self.expdic), 'explanations')

        # Reading vocabulary dictionaries
        with open(adict_path,'r') as f:
            self.adict = json.load(f)
        with open(exp_vdict_path,'r') as f:
            self.exp_vdict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)
        self.n_exp_vocabulary = len(self.exp_vdict)

    def getQuesIds(self):
        return list(self.qdic.keys())

    def getImgId(self,qid):
        return self.qdic[qid]

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    def getExpStr(self, qid):
        return self.expdic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        return q_list


    def exp_list_to_vec(self, max_length, e_list):
        seq_length = len(e_list)
        pad_length = max(0, max_length - seq_length -1)
        exp_list = []
        for w in e_list:
            if w not in self.exp_vdict:
                w = ''
            exp_list.append(self.exp_vdict[w])

        exp_cont_1 = [0] + seq_length * [1] + pad_length * [0]
        exp_cont_2 = [0] + seq_length * [1] + pad_length * [0]
        exp_cont_1 = exp_cont_1[:max_length]
        exp_cont_2 = exp_cont_1[:max_length]

        sos = [0]
        eos = [0]
        exp_in = sos + exp_list[:] + pad_length * [0]
        exp_out = exp_list[:] + eos + pad_length * [-1]
        exp_in = exp_in[:max_length]
        exp_out = exp_out[:max_length]

        exp_vec = np.array(exp_in)
        exp_vec_out = np.array(exp_out)
        exp_cvec_1 = np.array(exp_cont_1)
        exp_cvec_2 = np.array(exp_cont_2)

        return exp_vec, exp_vec_out, exp_cvec_1, exp_cvec_2

 
    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        if ans_str in self.adict:
            ans = self.adict[ans_str]
        else:
            ans = -1
        return ans
 
    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]
 
    def create_batch(self,qid_list):

        ivec = np.zeros((self.batchsize, self.data_shape[0], self.data_shape[1], self.data_shape[2]))
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        exp_vec = np.zeros((self.batchsize, self.exp_max_length))
        exp_vec_out = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_1 = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_2 = np.zeros((self.batchsize, self.exp_max_length))

        for i,qid in enumerate(qid_list):

            # load raw information
            q_ans_str = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)
            exp_str = self.getExpStr(qid)

            # convert explanation to vec
            if isinstance(exp_str, str):
                exp_list = ActivityDataProvider.seq_to_list(exp_str)
                t_exp_vec, t_exp_vec_out, t_exp_cvec_1, t_exp_cvec_2 = \
                    self.exp_list_to_vec(self.exp_max_length, exp_list)
            # For validation set, we have 3 explanations per question
            else:
                exp_list = ActivityDataProvider.seq_to_list(exp_str[0])
                t_exp_vec, t_exp_vec_out, t_exp_cvec_1, t_exp_cvec_2 = \
                    self.exp_list_to_vec(self.exp_max_length, exp_list)

            try:
                t_ivec = np.load(self.img_feature_prefix + str(q_iid) + '.jpg.npz')['x']
                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                print('data not found for qid : ', qid,  self.mode)
             
            # convert answer to vec
            t_avec = self.answer_to_vec(q_ans_str)

            ivec[i,...] = t_ivec
            avec[i,...] = t_avec
            exp_vec[i,...] = t_exp_vec
            exp_vec_out[i,...] = t_exp_vec_out
            exp_cvec_1[i,...] = t_exp_cvec_1
            exp_cvec_2[i,...] = t_exp_cvec_2

        return ivec, avec, exp_vec, exp_vec_out, exp_cvec_1, exp_cvec_2

    def get_batch_vec(self):
        if self.batch_len is None:
            qid_list = self.getQuesIds()
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            t_qid_list.append(t_qid)
            t_iid_list.append(t_iid)
            counter += 1

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                self.qid_list = qid_list
                self.batch_index = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class ActivityDataProviderLayer(caffe.Layer):
    """
    Provide input data for Activity classification.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        self.top_names = ['feature', 'label', 'exp', 'exp_out', 'exp_cont_1', 'exp_cont_2']
        max_exp_words = MAX_WORDS_IN_EXP
        top[0].reshape(self.batchsize, *CURRENT_DATA_SHAPE)
        top[1].reshape(self.batchsize)
        top[2].reshape(max_exp_words, self.batchsize)
        top[3].reshape(max_exp_words, self.batchsize)
        top[4].reshape(max_exp_words, self.batchsize)
        top[5].reshape(max_exp_words, self.batchsize)

        self.mode = json.loads(self.param_str)['mode']
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            raise NotImplementedError

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            raise NotImplementedError

    def backward(self, top, propagate_down, bottom):
        pass

