GPU_ID = 1
BATCH_SIZE = 64 
VAL_BATCH_SIZE = 104
NUM_OUTPUT_UNITS = 397 # This is the answer vocabulary size
MAX_WORDS_IN_EXP = 36
MAX_ITERATIONS = 40000
PRINT_INTERVAL = 100

# what data to use for training
TRAIN_DATA_SPLITS = 'train'

# what data to use for the vocabulary
ANSWER_VOCAB_SPACE = 'train'
EXP_VOCAB_SPACE = 'train'

# location of the data
ACTIVITY_PREFIX = './ACT-X'

DATA_PATHS = {
	'train': {
		'ans_file': ACTIVITY_PREFIX + '/textual/exp_train_split.json',
		'features_prefix': ACTIVITY_PREFIX + '/Features/resnet_res5c_bgrms_large/'
	},
	'val': {
		'ans_file': ACTIVITY_PREFIX + '/textual/exp_val_split.json',
		'features_prefix': ACTIVITY_PREFIX + '/Features/resnet_res5c_bgrms_large/'
	}
}
