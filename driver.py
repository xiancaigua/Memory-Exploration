import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random

from parameter import *
from model import PolicyNet, QNet, NeuralTuringMachine
from runner import RLRunner
# tune the GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# python driver.py > loggings/log04-11-ntm1-ntm.txt 2>&1 &

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)
map_data_folder = "maps_medium_data"
if not os.path.exists(f'{map_data_folder}'):
            os.makedirs(f'{map_data_folder}')


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')

    # initialize neural networks
    global_policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net1 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_q_net2 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)
    global_target_q_net2 = QNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM).to(device)

    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
    
    # target entropy for SAC
    entropy_target = 0.1 * (-np.log(1 / LOCAL_K_SIZE))

    curr_episode = 0
    target_q_update_counter = 1

    if EXPERIMENT_MODE == 'ntm':
        neural_turing_machine = NeuralTuringMachine(
            input_dim=EMBEDDING_DIM,
            output_dim=EMBEDDING_DIM,
            memory_size=20,
            memory_dim=EMBEDDING_DIM,
            num_heads=2,
            batch_size=BATCH_SIZE,
            device=device
        )
    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        log_alpha = checkpoint['log_alpha']  # not trainable when loaded from checkpoint, manually tune it for now
        log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)
        # print("log_alpha", log_alpha.device)
        if log_alpha.device != device:
            log_alpha = log_alpha.to(device)
            # print("log_alpha", log_alpha.device)

        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha, log_alpha.requires_grad)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])
    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()
    
    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)
    
    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        # job_list.append(meta_agent.job.remote(weights_set, curr_episode))
        job_list.append(meta_agent.job.remote(weights_set, curr_episode, neural_turing_machine))

    # initialize metric collector
    metric_name = ['travel_dist', 'success_rate', 'explored_rate']
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    ground_truth_experience_buffer = []
    for _ in range(19):
        experience_buffer.append([])
    for _ in range(12):
        ground_truth_experience_buffer.append([])

    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)

            # save experience and metric
            for job in done_jobs:
                # job_results, metrics, info = job
                job_results, ground_truth_job_result, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                    
                for i in range(len(ground_truth_experience_buffer)):
                    ground_truth_experience_buffer[i] += ground_truth_job_result[i]
                    
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            # job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))

            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training-------------------------episode", curr_episode)

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]
                    
                    for i in range(len(ground_truth_experience_buffer)):
                        ground_truth_experience_buffer[i] = ground_truth_experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(4):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])
                        
                    for i in range(len(ground_truth_experience_buffer)):
                        rollouts.append([ground_truth_experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    local_node_inputs = torch.stack(rollouts[0]).to(device)
                    local_node_padding_mask = torch.stack(rollouts[1]).to(device)
                    local_edge_mask = torch.stack(rollouts[2]).to(device)
                    current_local_index = torch.stack(rollouts[3]).to(device)
                    current_local_edge = torch.stack(rollouts[4]).to(device)
                    local_edge_padding_mask = torch.stack(rollouts[5]).to(device)
                    robot_location = torch.stack(rollouts[6]).to(device)
                    stacked_msgs = torch.stack(rollouts[7]).to(device)
                    action = torch.stack(rollouts[8]).to(device)
                    reward = torch.stack(rollouts[9]).to(device)
                    done = torch.stack(rollouts[10]).to(device)
                    next_local_node_inputs = torch.stack(rollouts[11]).to(device)
                    next_local_node_padding_mask = torch.stack(rollouts[12]).to(device)
                    next_local_edge_mask = torch.stack(rollouts[13]).to(device)
                    next_current_local_index = torch.stack(rollouts[14]).to(device)
                    next_current_local_edge = torch.stack(rollouts[15]).to(device)
                    next_local_edge_padding_mask = torch.stack(rollouts[16]).to(device)
                    next_robot_location = torch.stack(rollouts[17]).to(device)
                    next_stacked_msgs = torch.stack(rollouts[18]).to(device)
                    
                    ground_truth_local_node_inputs = torch.stack(rollouts[19]).to(device)
                    ground_truth_local_node_padding_mask = torch.stack(rollouts[20]).to(device)
                    ground_truth_local_edge_mask = torch.stack(rollouts[21]).to(device)
                    ground_truth_current_local_index = torch.stack(rollouts[22]).to(device)
                    ground_truth_current_local_edge = torch.stack(rollouts[23]).to(device)
                    ground_truth_local_edge_padding_mask = torch.stack(rollouts[24]).to(device)
                    
                    ground_truth_next_local_node_inputs = torch.stack(rollouts[25]).to(device)
                    ground_truth_next_local_node_padding_mask = torch.stack(rollouts[26]).to(device)
                    ground_truth_next_local_edge_mask = torch.stack(rollouts[27]).to(device)
                    ground_truth_next_current_local_index = torch.stack(rollouts[28]).to(device)
                    ground_truth_next_current_local_edge = torch.stack(rollouts[29]).to(device)
                    ground_truth_next_local_edge_padding_mask = torch.stack(rollouts[30]).to(device)
                    

                    observation = [local_node_inputs, local_node_padding_mask, local_edge_mask, current_local_index,
                                   current_local_edge, local_edge_padding_mask, robot_location, stacked_msgs]
                    next_observation = [next_local_node_inputs, next_local_node_padding_mask, next_local_edge_mask,
                                        next_current_local_index, next_current_local_edge, next_local_edge_padding_mask,next_robot_location, next_stacked_msgs]
                    
                    ground_truth_obersevation = [ground_truth_local_node_inputs, ground_truth_local_node_padding_mask, ground_truth_local_edge_mask, ground_truth_current_local_index,
                                      ground_truth_current_local_edge, ground_truth_local_edge_padding_mask]
                    
                    ground_truth_next_observation = [ground_truth_next_local_node_inputs, ground_truth_next_local_node_padding_mask, ground_truth_next_local_edge_mask,
                                        ground_truth_next_current_local_index, ground_truth_next_current_local_edge, ground_truth_next_local_edge_padding_mask]
                    # SAC

                    with torch.no_grad():
                        q_values1 = dp_q_net1(*ground_truth_obersevation)
                        q_values2 = dp_q_net2(*ground_truth_obersevation)
                        q_values = torch.min(q_values1, q_values2)
                        
                    logp = dp_policy(*observation)

                    # print("logp", logp.device)
                    # print("q_values", q_values.device)
                    # print("log_alpha", log_alpha.device)
                    policy_loss = torch.sum(
                        (logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())),
                        dim=1).mean()
                    
                    policy_loss = policy_loss
                    
                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100,
                                                                    norm_type=2)
                    global_policy_optimizer.step()


                    with torch.no_grad():
                        next_logp = dp_policy(*next_observation)
                        next_q_values1 = dp_target_q_net1(*ground_truth_next_observation)
                        next_q_values2 = dp_target_q_net2(*ground_truth_next_observation)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime = torch.sum(
                            next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)),
                            dim=1).unsqueeze(1)
                        target_q_batch = reward + GAMMA * (1 - done) * value_prime
                        

                    mse_loss = nn.MSELoss()
                    q_values1  = dp_q_net1(*ground_truth_obersevation)
                    q1 = torch.gather(q_values1, 1, action)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    q1_loss = q1_loss

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net1_optimizer.step()

                    q_values2 = dp_q_net2(*ground_truth_obersevation)
                    q2 = torch.gather(q_values2, 1, action)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()
                    
                    q2_loss = q2_loss

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000,
                                                                 norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()
                    

                    target_q_update_counter += 1
                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward.mean().item(), value_prime.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(),
                        alpha_loss.item(), *perf_data]
                
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                write_to_tensor_board(writer, training_data, curr_episode)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                global_policy_net.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
            weights_set.append(policy_weights)

            # update the target q net
            if target_q_update_counter > 128:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            if curr_episode % SAVE_WINDOW == 0:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                              "q_net1_model": global_q_net1.state_dict(),
                              "q_net2_model": global_q_net2.state_dict(),
                              "log_alpha": log_alpha,
                              "policy_optimizer": global_policy_optimizer.state_dict(),
                              "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                              "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                              "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                              "episode": curr_episode,
                              }
                path_checkpoint = "./" + model_path + "/{}_checkpoint.pth".format(curr_episode)
                torch.save(checkpoint, path_checkpoint)
                # print('Saved model', end='\n')
                print('Model saved to', path_checkpoint, end='\n')

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)


def write_to_tensor_board(writer, tensorboard_data, curr_episode):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboard_data = np.array(tensorboard_data)
    tensorboard_data = list(np.nanmean(tensorboard_data, axis=0))
    reward, value, policy_loss, q_value_loss, entropy, policy_grad_norm, q_value_grad_norm, log_alpha, alpha_loss, travel_dist, success_rate, explored_rate = tensorboard_data
    
    writer.add_scalar(tag='Losses/Value', scalar_value=value, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Loss', scalar_value=policy_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Alpha Loss', scalar_value=alpha_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Loss', scalar_value=q_value_loss, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Entropy', scalar_value=entropy, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Policy Grad Norm', scalar_value=policy_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Q Value Grad Norm', scalar_value=q_value_grad_norm, global_step=curr_episode)
    writer.add_scalar(tag='Losses/Log Alpha', scalar_value=log_alpha, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Reward', scalar_value=reward, global_step=curr_episode)
    
    writer.add_scalar(tag='Perf/Travel Distance', scalar_value=travel_dist, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Explored Rate', scalar_value=explored_rate, global_step=curr_episode)
    writer.add_scalar(tag='Perf/Success Rate', scalar_value=success_rate, global_step=curr_episode)


if __name__ == "__main__":
    main()
