import ray
import numpy as np
import torch
from test_worker import TestWorker
from test_parameter import *
from model import PolicyNet,NeuralTuringMachine
import os

def run_test(param=300):
    global gifs_path
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, mode=mode).to(device)
    if mode == 'ntm':
        pretrained_path = "model/memory/pretrained_ntm.pth"
        neural_turing_machine = NeuralTuringMachine(
            input_size=3 * LOCAL_NODE_INPUT_DIM,
            output_size=3 * LOCAL_NODE_INPUT_DIM,
            controller_size=80,
            memory_size=NTM_SIZE,
            memory_vector_size=EMBEDDING_DIM,
            num_heads=2,
            batch_size=1,
            train_device=device,
            work_device=device
        )
        checkpoint = torch.load(pretrained_path, map_location=device)
    else:
        neural_turing_machine = None
    
    model_path_ = f'{model_path}/{param}_checkpoint.pth'
    # model_path_ = 'model/ntm-9/1_checkpoint.pth'
    if device == 'cuda':
        checkpoint = torch.load(model_path_)
    else:
        checkpoint = torch.load(model_path_, map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])
    if mode == 'ntm':
        neural_turing_machine.load_state_dict(checkpoint['neural_turing_machine'])
        memory = checkpoint['memory']
        neural_turing_machine.update_memory(memory)
    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    # meta_agents = Runner(0) #debug
    weights = global_network.state_dict()
    curr_test = 0
    over_test = 0

    dist_historys = [[],[],[]]
    successes = [[],[],[]]

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_test += 1
        job_list.append(meta_agent.job.remote(weights, curr_test,neural_turing_machine,gifs_path))
    # curr_test += 1 #debug
    try:
        while True:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            # done_jobs = [meta_agents.job(weights,curr_test,neural_turing_machine,gifs_path)] #debug
            for job in done_jobs:
                over_test += len(done_jobs)
                metrics, info = job
                index =((info['episode_number'] -1) // (NUM_TEST // 3)) % 3
                dist_historys[index].append(metrics['travel_dist'])
                successes[index].append(metrics['success_rate'])
            if curr_test < NUM_TEST:
                curr_test += 1
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test,neural_turing_machine,gifs_path)) #debug
            for i in range(len(dist_historys)):
                dist_history = dist_historys[i]
                success = successes[i]
                if len(dist_history) == (NUM_TEST // 3):
                    avg_length = np.array(dist_history).mean()
                    length_std = np.array(dist_history).std()
                    success_rate = float(np.array(success).sum()) / (NUM_TEST // 3)

                    print(f'Finished {over_test} tests')
                    print(f'|#Scenario {i} test')
                    print('|#Total test:', len(dist_history))
                    print('|#Average length:', avg_length)
                    print('|#Length std:', length_std)
                    print('|#Success rate', success_rate)
                    file_path = result_path + '/' + "test_results.txt"
                    with open(file_path, "a") as file:
                        file.write(f'*{param}* checkpoint Finished {over_test} tests\n')
                        file.write(f'|#Scenario {i} test\n')
                        file.write(f'|#Total test: {len(dist_history)}\n')
                        file.write(f'|#Average length: {avg_length}\n')
                        file.write(f'|#Length std: {length_std}\n')
                        file.write(f'|#Success rate: {success_rate}\n')
                        file.write('\n')  # 添加空行以分隔不同测试结果
                    # 对于某个场景清空表现
                    dist_historys[i] = []
                    successes[i] = []
            
            if over_test >= NUM_TEST:
                print("All tests finished")
                break
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

def run_test_greed(param=300):
    global gifs_path
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, mode=mode).to(device)
    neural_turing_machine = None
    model_path_ = '/home/zzh/Memory-Exploration/model/ex7/100_checkpoint.pth'
    
    if device == 'cuda':
        checkpoint = torch.load(model_path_)
    else:
        checkpoint = torch.load(model_path_, map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i) for i in range(NUM_META_AGENT)]
    # meta_agents = Runner(0) #debug
    weights = global_network.state_dict()
    curr_test = 0
    over_test = 0

    dist_historys = [[],[],[]]
    successes = [[],[],[]]

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_test += 1
        job_list.append(meta_agent.job.remote(weights, curr_test,neural_turing_machine,gifs_path))
    # curr_test += 1 #debug
    try:
        while True:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)
            # done_jobs = [meta_agents.job(weights,curr_test,neural_turing_machine,gifs_path)] #debug
            for job in done_jobs:
                over_test += len(done_jobs)
                metrics, info = job
                index =((info['episode_number'] -1) // (NUM_TEST // 3)) % 3
                dist_historys[index].append(metrics['travel_dist'])
                successes[index].append(metrics['success_rate'])
            if curr_test < NUM_TEST:
                curr_test += 1
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test,neural_turing_machine,gifs_path)) #debug
            for i in range(len(dist_historys)):
                dist_history = dist_historys[i]
                success = successes[i]
                if len(dist_history) == (NUM_TEST // 3):
                    avg_length = np.array(dist_history).mean()
                    length_std = np.array(dist_history).std()
                    success_rate = float(np.array(success).sum()) / (NUM_TEST // 3)

                    print(f'Finished {over_test} tests')
                    print(f'|#Scenario {i} test')
                    print('|#Total test:', len(dist_history))
                    print('|#Average length:', avg_length)
                    print('|#Length std:', length_std)
                    print('|#Success rate', success_rate)
                    file_path = result_path + '/' + "test_results.txt"
                    with open(file_path, "a") as file:
                        file.write(f'*{param}* checkpoint Finished {over_test} tests\n')
                        file.write(f'|#Scenario {i} test\n')
                        file.write(f'|#Total test: {len(dist_history)}\n')
                        file.write(f'|#Average length: {avg_length}\n')
                        file.write(f'|#Length std: {length_std}\n')
                        file.write(f'|#Success rate: {success_rate}\n')
                        file.write('\n')  # 添加空行以分隔不同测试结果
                    # 对于某个场景清空表现
                    dist_historys[i] = []
                    successes[i] = []
            
            if over_test >= NUM_TEST:
                print("All tests finished")
                break
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)

@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT) #debug
class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM, mode=mode)
        self.local_network.to(self.device)
        self.neural_turing_machine = None

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number, gifs_path):
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, neural_turing_machine=self.neural_turing_machine ,device=self.device, save_image=SAVE_GIFS, greedy=True, gifs_path = gifs_path)
        if mode == 'greed':
            worker.run_greed_episode()
        else:
            worker.run_episode()

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number, neural_turing_machine, gifs_path):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)
        self.neural_turing_machine = neural_turing_machine
        metrics = self.do_job(episode_number, gifs_path)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info

if __name__ == '__main__':
        # ray.init(object_store_memory=2 * 1024 * 1024 * 1024)  # 限制为 2GB    for i in range(NUM_RUN):
        test = [300, 600, 900]
        for i in test:
            gifs_path_ = result_path + '/' + 'gifs'
            gifs_path = gifs_path_ + '_' + str(i)
            if not os.path.exists(gifs_path):
                os.makedirs(gifs_path)
            if mode == 'greed':
                run_test_greed(i)
            else:
                run_test(i) 
        # run_test()
        # CUDA_VISIBLE_DEVICES=3 python test_driver.py > results/ntm-13/log.txt 2>&1