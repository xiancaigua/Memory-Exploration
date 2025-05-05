FOLDER_NAME = 'ntm-13'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'
SUMMARY_WINDOW = 32
SAVE_WINDOW = 100
LOAD_MODEL = True  # do you want to load the model trained before
SAVE_IMG_GAP = 100  # 100
EXPERIMENT_MODE = 'ntm' # code of experiment
USE_MULTI_THREAD = True  # do you want to use multi-threading
# base ; ntm ; gen ; origin

N_AGENTS = 4
NUM_META_AGENT = 10
#########################################
# Continual Learning parameters
REPLAY_TIMES = 300
TOTAL_SCENARIO = 3
NTM_SIZE = 128
TRUNC_VALUE = 2000
#########################################
# Environment parameters

CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0 # meter
DOWNSAMPLE_SIZE = NODE_RESOLUTION // CELL_SIZE

SENSOR_RANGE = 20  # meter  20
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

LOCAL_MAP_SIZE = 40  # meter  40-20
EXTENDED_LOCAL_MAP_SIZE = 6 * SENSOR_RANGE * 1.05

#########################################
# Learning parameters

PRIVILEDGED_INFO = True  # do you want to use privileged information
MAX_EPISODE_STEP = 128 # 128
REPLAY_SIZE = 5000   # 10000
MINIMUM_BUFFER_SIZE = 2500 # 5000
BATCH_SIZE = 256  # 128
LR = 2e-4  # 1e-5
GAMMA = 1

LOCAL_NODE_INPUT_DIM = 5
OTHER_INFO_INPUT_DIM = 3
EMBEDDING_DIM = 16

LOCAL_K_SIZE = 25  # the number of neighboring nodes
LOCAL_NODE_PADDING_SIZE = 420  # the number of nodes will be padded to this value  360

USE_GPU = True  # do you want to collect training data using GPUs  NOT RECOMMENDED!!!
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1