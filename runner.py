import torch
import ray
from model import PolicyNet
from multi_agent_worker import Multi_agent_worker
from parameter import *
import sys

# Runner 类用于管理强化学习任务的执行，包括策略网络的加载和任务分配。
class Runner(object):
    def __init__(self, meta_agent_id):
        # 初始化 Runner，包括元代理 ID 和设备设置。
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM, mode=EXPERIMENT_MODE)
        self.local_network.to(self.device)
        self.neural_turing_machine = None

    def get_weights(self):
        # 获取当前策略网络的权重。
        return self.local_network.state_dict()
    
    def set_policy_net_weights(self, weights):
        # 设置策略网络的权重。
        self.local_network.load_state_dict(weights)
    
    def do_job(self, episode_number, in_pretrain=False):
        # 执行一个任务周期，包括环境交互和性能指标计算。
        current_memory = None

        if in_pretrain:
            save_img = False
        else:
            save_img = True if ((episode_number % SAVE_IMG_GAP == 0) or \
                                ((episode_number-1) % SAVE_IMG_GAP == 0) \
                                    or ((episode_number-2) % SAVE_IMG_GAP == 0))  else False

        worker = Multi_agent_worker(self.meta_agent_id, self.local_network, episode_number, self.neural_turing_machine, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        ground_truth_job_results = worker.ground_truth_episode_buffer
        perf_metrics = worker.perf_metrics

        if EXPERIMENT_MODE == 'ntm':
            current_memory = worker.ntm.get_memory()
        return job_results, ground_truth_job_results, perf_metrics, current_memory
    
    def job(self, weights_set, episode_number, neural_turing_machine, in_pretrain=False):
        # 分配任务并返回结果，包括任务数据和性能指标。
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_policy_net_weights(weights_set[0])
        self.neural_turing_machine = neural_turing_machine

        job_results, ground_truth_job_results, metrics, current_memory = self.do_job(episode_number,in_pretrain)

        print("finished episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))

        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        
        # print(f"Size of job_results: {sys.getsizeof(job_results)} bytes")
        # print(f"Size of ground_truth_job_results: {sys.getsizeof(ground_truth_job_results)} bytes")
        # print(f"Size of metrics: {sys.getsizeof(metrics)} bytes")
        
        return job_results, ground_truth_job_results, metrics, current_memory, info


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(47)
    out = ray.get(job_id)
    print(out[1])
