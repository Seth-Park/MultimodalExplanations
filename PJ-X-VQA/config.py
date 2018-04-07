GPU_ID = 0
BATCH_SIZE = 64 
VAL_BATCH_SIZE = 100
NUM_OUTPUT_UNITS = 3000 # This is the answer vocabulary size
MAX_WORDS_IN_QUESTION = 15
MAX_WORDS_IN_EXP = 36
MAX_ITERATIONS = 50000
PRINT_INTERVAL = 100

# what data to use for training
TRAIN_DATA_SPLITS = 'train'

# what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train'
ANSWER_VOCAB_SPACE = 'train'
EXP_VOCAB_SPACE = 'train'

# VQA pretrained model
VQA_PRETRAINED = '/data/seth/snapshots/VQA2/_iter_195000.caffemodel'
#VQA_PRETRAINED = 'PATH_TO_PRETRAINED_VQA_MODEL.caffemodel'

# location of the data
VQA_PREFIX = './VQA-X'

DATA_PATHS = {
	'train': {
		'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_train2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/v2_mscoco_train2014_annotations.json',
		'exp_file': VQA_PREFIX + '/Annotations/train_exp_anno.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_res5c_bgrms_large/train2014/COCO_train2014_'
	},
	'val': {
		'ques_file': VQA_PREFIX + '/Questions/v2_OpenEnded_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/v2_mscoco_val2014_annotations.json',
		'exp_file': VQA_PREFIX + '/Annotations/val_exp_anno.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_res5c_bgrms_large/val2014/COCO_val2014_'
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	},
	'test': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
		'features_prefix': VQA_PREFIX + '/Features/resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	}
}
