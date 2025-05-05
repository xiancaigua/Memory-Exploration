TEST_N_AGENTS = 4  # the number of agents in the test 如果数量大于4不要可视化！
mode = 'ntm'  # code of experiment
# base ; ntm ; gen ; origin ; greed

NODE_INPUT_DIM = 5
OTHER_INFO_INPUT_DIM = 3
LOCAL_NODE_INPUT_DIM = 5
EMBEDDING_DIM = 16
K_SIZE = 25  # the number of neighbors

USE_GPU = True  # do you want to use GPUS?
NUM_GPU = 1
NUM_META_AGENT = 10  # the number of processes
FOLDER_NAME = 'ntm-13'
if mode == 'greed':
    K_SIZE = 10
    FOLDER_NAME = 'greed'

model_path = f'model/{FOLDER_NAME}'
result_path = f'results/{FOLDER_NAME}'

NUM_TEST = 300
NUM_RUN = 1
SAVE_GIFS = True  # do you want to save GIFs
NTM_SIZE = 128