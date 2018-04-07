import caffe
import numpy as np
import re, json, random
import config

QID_KEY_SEPARATOR = '/'

class ActivityDataProvider:

    def __init__(self, batchsize=64,
                 exp_max_length=config.MAX_WORDS_IN_EXP, mode='train'):
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.exp_max_length = exp_max_length
        self.mode = mode
        self.qdic, self.adic, self.expdic = ActivityDataProvider.load_data(mode)

        with open('./model/adict.json','r') as f:
            self.adict = json.load(f)
        with open('./model/exp_vdict.json','r') as f:
            self.exp_vdict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)

    @staticmethod
    def load_activity_json(data_split):
        """
        Parses answer and explanation json file for the given data split. 
        Returns the answer dictionary and explanation dictionary.
        """
        qdic, adic, exp_dic = {}, {}, {}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'], 'r') as f:
                adata = json.load(f)
                for k, v in adata.items():
                    qdic[k] = v['iid']
                    adic[k] = v['ans']
                    exp_dic[k] = v['exp']

        print('parsed', len(qdic), 'questions for', data_split)
        print('parsed', len(exp_dic), 'explanations for', data_split)
        
        return qdic, adic, exp_dic

    @staticmethod
    def load_data(data_split_str):
        all_qdic, all_adic, all_expdic = {}, {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            qdic, adic, expdic = ActivityDataProvider.load_activity_json(data_split)
            all_qdic.update(qdic)
            all_adic.update(adic)
            all_expdic.update(expdic)
        return all_qdic, all_adic, all_expdic


    def getQuesIds(self):
        return list(self.qdic.keys())

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    def getExpStr(self, qid):
        exp_str = self.expdic[qid]
        if isinstance(exp_str, str):
            return exp_str
        else:
            return random.choice(exp_str)

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
            ans = self.adict['']
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

        ivec = (np.zeros(self.batchsize*2048*14*14)).reshape(self.batchsize,2048,14,14)
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        exp_vec = np.zeros((self.batchsize, self.exp_max_length))
        exp_vec_out = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_1 = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_2 = np.zeros((self.batchsize, self.exp_max_length))

        for i,qid in enumerate(qid_list):

            if qid[0] == 't':
                data_split = 'train'
            else:
                data_split = 'val'
            # load raw question information
            q_ans_str = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)
            exp_str = self.getExpStr(qid)

            # convert explanation to vec
            exp_list = ActivityDataProvider.seq_to_list(exp_str)
            t_exp_vec, t_exp_vec_out, t_exp_cvec_1, t_exp_cvec_2 = \
                self.exp_list_to_vec(self.exp_max_length, exp_list)

            try:
                t_ivec = np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid) + '.jpg.npz')['x']
                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                print('data not found for qid : ', q_iid,  self.mode)
             
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
            random.shuffle(qid_list)
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
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class ActivityDataProviderLayer(caffe.Layer):
    """
    Provide input data for VQA.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        self.top_names = ['feature', 'label', 'exp', 'exp_out', 'exp_cont_1', 'exp_cont_2']
        max_exp_words = config.MAX_WORDS_IN_EXP
        top[0].reshape(self.batchsize,2048,14,14)
        top[1].reshape(self.batchsize)
        top[2].reshape(max_exp_words, self.batchsize)
        top[3].reshape(max_exp_words, self.batchsize)
        top[4].reshape(max_exp_words, self.batchsize)
        top[5].reshape(max_exp_words, self.batchsize)

        self.mode = json.loads(self.param_str)['mode']
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            self.dp = ActivityDataProvider(batchsize=self.batchsize, mode=self.mode)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            feature, answer, exp, exp_out, exp_cont_1, exp_cont_2, _, _, _ = self.dp.get_batch_vec()
            top[0].data[...] = feature
            top[1].data[...] = answer
            top[2].data[...] = np.transpose(exp, (1, 0))
            top[3].data[...] = np.transpose(exp_out, (1, 0))
            top[4].data[...] = np.transpose(exp_cont_1, (1, 0))
            top[5].data[...] = np.transpose(exp_cont_2, (1, 0))

    def backward(self, top, propagate_down, bottom):
        pass

