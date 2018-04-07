import caffe
import numpy as np
import re, json, random
import config

QID_KEY_SEPARATOR = '/'

class VQADataProvider:

    def __init__(self, batchsize=64, max_length=config.MAX_WORDS_IN_QUESTION,
                 exp_max_length=config.MAX_WORDS_IN_EXP, mode='train'):
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.exp_max_length = exp_max_length
        self.mode = mode
        self.qdic, self.adic, self.expdic = VQADataProvider.load_data(mode)
        self.qdic, self.adic, self.expdic = VQADataProvider.filter_for_exp(self.qdic,
                                                                           self.adic,
                                                                           self.expdic)

        with open('./model/vdict.json','r') as f:
            self.vdict = json.load(f)
        with open('./model/adict.json','r') as f:
            self.adict = json.load(f)
        with open('./model/exp_vdict.json','r') as f:
            self.exp_vdict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)

    @staticmethod
    def load_vqa_json(data_split):
        """
        Parses the question and answer and explanation json files for the given data split. 
        Returns the question dictionary, answer dictionary, and explanation dictionary.
        """
        qdic, adic, exp_dic = {}, {}, {}

        with open(config.DATA_PATHS[data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
                    {'qstr': q['question'], 'iid': q['image_id']}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
                    adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
                        a['answers']
            with open(config.DATA_PATHS[data_split]['exp_file'], 'r') as f:
                expdata = json.load(f)
                for qid, exp in expdata.items():
                    exp_dic[data_split + QID_KEY_SEPARATOR + str(qid)] = exp

        print('parsed', len(qdic), 'questions for', data_split)
        print('parsed', len(exp_dic), 'explanations for', data_split)
        
        return qdic, adic, exp_dic

    @staticmethod
    def load_data(data_split_str):
        all_qdic, all_adic, all_expdic = {}, {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            qdic, adic, expdic = VQADataProvider.load_vqa_json(data_split)
            all_qdic.update(qdic)
            all_adic.update(adic)
            all_expdic.update(expdic)
        return all_qdic, all_adic, all_expdic

    @staticmethod
    def filter_for_exp(qdic, adic, expdic):
        """
        Get rid of QA pairs that does not have explanation labels.
        """
        filtered_qdic = {}
        filtered_adic = {}
        for qid in expdic.keys():
            qdata = qdic[qid]
            adata = adic[qid]
            filtered_qdic[qid] = qdata
            filtered_adic[qid] = adata
        assert len(filtered_qdic) == len(filtered_adic) == len(expdic)
        print('final training data has', len(expdic), 'questions')
        return filtered_qdic, filtered_adic, expdic

    def getQuesIds(self):
        return list(self.qdic.keys())

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    def getExpStr(self, qid):
        return np.random.choice(self.expdic[qid], size=1)[0]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        return q_list

    def extract_answer(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [ answer_obj[i]['answer'] for i in range(10)]
        dic = {}
        for ans in answer_list:
            if ans in dic:
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self,answer_obj):
        """ Randomly sample from possible set of answers."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if ans in self.adict:
                prob_answer_list.append(ans)

        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)
 
    def qlist_to_vec(self, max_length, q_list):
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        for i,_ in enumerate(range(max_length)):
            if i < max_length - len(q_list):
                cvec[i] = 0
            elif i == max_length - len(q_list):
                w = q_list[i-(max_length-len(q_list))]
                # is the word in the vocabulary?
                if w not in self.vdict:
                    w = ''
                qvec[i] = self.vdict[w]
                cvec[i] = 0
            else:
                w = q_list[i-(max_length-len(q_list))]
                # is the word in the vocabulary?
                if w not in self.vdict:
                    w = ''
                qvec[i] = self.vdict[w]
                cvec[i] = 1
        return qvec, cvec


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

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        ivec = (np.zeros(self.batchsize*2048*14*14)).reshape(self.batchsize,2048,14,14)
        avec = (np.zeros(self.batchsize)).reshape(self.batchsize)
        exp_vec = np.zeros((self.batchsize, self.exp_max_length))
        exp_vec_out = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_1 = np.zeros((self.batchsize, self.exp_max_length))
        exp_cvec_2 = np.zeros((self.batchsize, self.exp_max_length))

        for i,qid in enumerate(qid_list):

            # load raw question information
            q_str = self.getQuesStr(qid)
            q_ans = self.getAnsObj(qid)
            q_iid = self.getImgId(qid)
            exp_str = self.getExpStr(qid)

            # convert question to vec
            q_list = VQADataProvider.seq_to_list(q_str)
            t_qvec, t_cvec = self.qlist_to_vec(self.max_length, q_list)

            # convert explanation to vec
            exp_list = VQADataProvider.seq_to_list(exp_str)
            t_exp_vec, t_exp_vec_out, t_exp_cvec_1, t_exp_cvec_2 = \
                self.exp_list_to_vec(self.exp_max_length, exp_list)

            try:
                qid_split = qid.split(QID_KEY_SEPARATOR)
                data_split = qid_split[0]
                if data_split == 'genome':
                    t_ivec = np.load(config.DATA_PATHS['genome']['features_prefix'] + str(q_iid) + '.jpg.npz')['x']
                else:
                    t_ivec = np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid).zfill(12) + '.jpg.npz')['x']
                t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
            except:
                t_ivec = 0.
                print('data not found for qid : ', q_iid,  self.mode)
             
            # convert answer to vec
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                q_ans_str = self.extract_answer(q_ans)
            else:
                q_ans_str = self.extract_answer_prob(q_ans)
            t_avec = self.answer_to_vec(q_ans_str)

            qvec[i,...] = t_qvec
            cvec[i,...] = t_cvec
            ivec[i,...] = t_ivec
            avec[i,...] = t_avec
            exp_vec[i,...] = t_exp_vec
            exp_vec_out[i,...] = t_exp_vec_out
            exp_cvec_1[i,...] = t_exp_cvec_1
            exp_cvec_2[i,...] = t_exp_cvec_2

        return qvec, cvec, ivec, avec, exp_vec, exp_vec_out, exp_cvec_1, exp_cvec_2

    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if ans in self.adict:
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1 

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)


class VQADataProviderLayer(caffe.Layer):
    """
    Provide input data for VQA.
    """

    def setup(self, bottom, top):
        self.batchsize = json.loads(self.param_str)['batchsize']
        self.top_names = ['data','cont','feature', 'label', 'exp', 'exp_out', 'exp_cont_1', 'exp_cont_2']
        max_ques_words = config.MAX_WORDS_IN_QUESTION
        max_exp_words = config.MAX_WORDS_IN_EXP
        top[0].reshape(max_ques_words,self.batchsize)
        top[1].reshape(max_ques_words,self.batchsize)
        top[2].reshape(self.batchsize,2048,14,14)
        top[3].reshape(self.batchsize)
        top[4].reshape(max_exp_words, self.batchsize)
        top[5].reshape(max_exp_words, self.batchsize)
        top[6].reshape(max_exp_words, self.batchsize)
        top[7].reshape(max_exp_words, self.batchsize)

        self.mode = json.loads(self.param_str)['mode']
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(batchsize=self.batchsize, mode=self.mode)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            word, cont, feature, answer, exp, exp_out, exp_cont_1, exp_cont_2, _, _, _ = self.dp.get_batch_vec()
            top[0].data[...] = np.transpose(word,(1,0))
            top[1].data[...] = np.transpose(cont,(1,0))
            top[2].data[...] = feature
            top[3].data[...] = answer
            top[4].data[...] = np.transpose(exp, (1, 0))
            top[5].data[...] = np.transpose(exp_out, (1, 0))
            top[6].data[...] = np.transpose(exp_cont_1, (1, 0))
            top[7].data[...] = np.transpose(exp_cont_2, (1, 0))

    def backward(self, top, propagate_down, bottom):
        pass

