import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ray
import random

from model import NeuralTuringMachine, PolicyNet
from runner import RLRunner
from parameter import *


ray.init()
torch.autograd.set_detect_anomaly(True)
# torch.cuda.set_device(2)

def generate_copy_data(seq_len=10, batch_size=1, input_size=8, device='cpu'):
    data = torch.rand(batch_size, seq_len, input_size).to(device)
    return data, data.clone()

def pretrain_ntm_model(
    input_size=3 * LOCAL_NODE_INPUT_DIM, output_size=3 * LOCAL_NODE_INPUT_DIM, controller_size=80,
    memory_size=128, memory_vector_size=EMBEDDING_DIM,
    num_heads=2, batch_size=1, sequence_length=400,
    epochs=10, lr=1e-3, save_every=50, save_path="model/memory/plot_.pth"
):
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    print(f"[INFO] Using device: {device}")


    ntm = NeuralTuringMachine(
            input_size=input_size,
            output_size=output_size,
            controller_size=controller_size,
            memory_size=memory_size,
            memory_vector_size=memory_vector_size,
            num_heads=num_heads,
            batch_size=batch_size,
            train_device=device,
            work_device=local_device
        ).to(device)

    print('Loading Action Model...')
    global_policy_net = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM,mode='base').to(device)
    checkpoint = torch.load("model/ex3/1100_checkpoint.pth", map_location=device)
    global_policy_net.load_state_dict(checkpoint['policy_model'])
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        global_policy_net.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
    weights_set.append(policy_weights)


    meta_agents = [RLRunner.remote(i) for i in range(3)]
    job_list = []
    curr_episode = 0
    for i, meta_agent in enumerate(meta_agents):
        # curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode, None, True))

    ntm.set_pretrain_mode(batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ntm.parameters(), lr=lr)

    loss_record = []
    data_buffer = []
    while True:

        done_id, job_list = ray.wait(job_list)
        done_jobs = ray.get(done_id)
        for job in done_jobs:
            job_results, ground_truth_job_result, metrics, current_memory_, info = job
            data_buffer += job_results[0]
        curr_episode += 1
        job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode, None, True))

        if len(data_buffer) < sequence_length:
            print("(T_T)数据不足，继续采集数据 in  episode :{curr_episode}...")
            continue
        elif len(data_buffer) > 8000:
            data_buffer = data_buffer[-1000:]

        for epoch in range(1, epochs + 1):
            print("[<(￣︶￣)↗[GO!]Training]")
            ntm.set_pretrain_mode(batch_size)
            optimizer.zero_grad()

            sample_indices = random.sample(range(len(data_buffer)), sequence_length)
            rollouts=[data_buffer[index] for index in sample_indices]
            _inputs = torch.stack(rollouts).to(device)
            selected = _inputs[:, :3, :]
            inputs = selected.reshape(sequence_length, 1, -1)
            targets = inputs.clone()
            # inputs, targets = generate_copy_data(sequence_length, batch_size, input_size, device)

            outputs = []
            for t in range(sequence_length):
                out = ntm(inputs[t, :, :])
                outputs.append(out.unsqueeze(1))

            outputs = torch.cat(outputs, dim=0)  # [B, T, D]
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ntm.parameters(), max_norm=10)
            optimizer.step()

            loss_record.append(loss.item())

        print(f"Epoch {epochs * curr_episode:03d}, Loss: {loss.item():.6f}")
        
        if (epochs * curr_episode) % save_every == 0:
            torch.save(ntm.state_dict(), save_path)
            print(f"[✓] Model checkpoint saved to {save_path}")
        
        if loss_record[-1] < 8e-3:
            print(f"[✓] Training over!----------------------------->")
            break

    return ntm, loss_record

def plot_loss_curve(loss_record, save_path='ntm_loss_curve.png'):
    plt.figure()
    plt.plot(loss_record)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("NTM Pretraining Loss Curve")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Loss curve saved to {save_path}")

def generate_memory_evolution_gif(model, input_seq, save_path="model/memory/memory_evolution.gif", dpi=100, interval=500):
    """
    生成 memory slot 激活的随时间变化 GIF 动画
    - model: 训练好的 NTM 模型
    - input_seq: [1, T, D] 输入序列
    """
    model.set_work_mode()
    memory_snapshots = []

    with torch.no_grad():
        for t in range(input_seq.shape[1]):
            _ = model(input_seq[:, t, :])
            memory_snapshots.append(model.memory.detach().cpu().clone())

    memory_snapshots = torch.stack(memory_snapshots)  # [T, M, V]
    memory_norms = memory_snapshots.norm(dim=2)       # [T, M] 每个 slot 的激活强度

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlabel("Memory Slot")
    ax.set_ylabel("Activation (L2 Norm)")
    ax.set_title("Memory Slot Activation Over Time")
    ax.set_xlim(0, memory_norms.shape[1])
    ax.set_ylim(0, memory_norms.max().item() * 1.1)
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        x = list(range(memory_norms.shape[1]))
        y = memory_norms[frame].numpy()
        line.set_data(x, y)
        ax.set_title(f"Memory Activation at Time Step {frame+1}")
        return line,

    ani = animation.FuncAnimation(
        fig, update, frames=memory_norms.shape[0],
        init_func=init, blit=True, interval=interval
    )

    ani.save(save_path, writer='pillow', dpi=dpi)
    plt.close()
    print(f"[✓] 动态 memory 可视化 GIF 已保存：{save_path}")

if __name__ == "__main__":
    ntm, loss_record = pretrain_ntm_model()
    plot_loss_curve(loss_record)