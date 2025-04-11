import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from copy import deepcopy
import numpy as np

from sensor import sensor_work
from parameter import *
from test_parameter import *
from utils import *


# Env 类用于模拟机器人探索环境的过程，包括地图加载、信念地图更新和奖励计算。
class Env:
    def __init__(self, episode_index, plot=False, test=False):
        # 初始化环境，包括地图加载、机器人位置设置和信念地图初始化。
        # 设置全局变量 N_AGENTS，根据是否为测试模式选择机器人数量
        global N_AGENTS
        if test:
            N_AGENTS = TEST_N_AGENTS
        
        # 标记是否为测试模式
        self.test = test
        # 当前任务的索引
        self.episode_index = episode_index
        # 是否启用绘图功能
        self.plot = plot
        # 地图路径，稍后通过 import_ground_truth 方法设置
        self.map_path = None
        
        # 加载真实地图和机器人初始位置
        self.ground_truth, initial_cell = self.import_ground_truth(episode_index)
        # 地图的大小（以网格单元为单位）
        self.ground_truth_size = np.shape(self.ground_truth)
        # 每个网格单元的实际尺寸（以米为单位）
        self.cell_size = CELL_SIZE

        # 机器人对环境的初始信念地图，初始值为 127（未知区域）
        self.robot_belief = np.ones(self.ground_truth_size) * 127
        # 信念地图的原点坐标（以米为单位），基于机器人初始位置计算
        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)

        # 传感器的感知范围（以米为单位）
        self.sensor_range = SENSOR_RANGE
        # 探索进度，初始为 0
        self.explored_rate = 0
        # 标记任务是否完成
        self.done = False

        # 根据初始位置和传感器范围更新信念地图
        self.robot_belief = sensor_work(initial_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                        self.ground_truth)
        # 信念地图的元信息（如原点、尺寸等）
        self.belief_info = Map_info(self.robot_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        # 真实地图的元信息
        self.ground_truth_info = Map_info(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        # 随机选择机器人初始位置
        free, _ = get_local_node_coords(np.array([0.0, 0.0]), self.belief_info)
        choice = np.random.choice(free.shape[0], N_AGENTS, replace=False) # 返回一个大小为 N_AGENTS 的一维数组，其中包含从 0 到 free.shape[0] - 1 范围内的随机整数
        starts = free[choice]
        self.robot_locations = np.array(starts)

        # 更新信念地图，标记机器人感知范围内的区域为已探索
        robot_cells = get_cell_position_from_coords(self.robot_locations, self.belief_info)
        # TODO: 这里的机器人信念更新存在冗余
        for robot_cell in robot_cells:
            self.robot_belief = sensor_work(robot_cell, self.sensor_range / self.cell_size, self.robot_belief,
                                            self.ground_truth)
        # 保存信念地图的初始状态
        self.old_belief = deepcopy(self.robot_belief)
        # 获取全局前沿（未探索区域的边界）
        self.global_frontiers = get_frontier_in_map(self.belief_info)

        # 如果启用了绘图功能，初始化绘图帧文件列表
        if self.plot:
            self.frame_files = []


    def import_ground_truth(self, episode_index):
        # 加载真实地图并处理为可用格式。
        # 确定地图目录，根据是否为测试模式选择不同的目录
        map_dir = f'maps_medium'
        if self.test:
            map_dir = f'maps_test'
        
        

        # 获取地图文件列表并选择当前任务对应的地图文件
        map_list = os.listdir(map_dir)
        if EXPERIMENT_MODE == 'origin':
            map_index = episode_index % np.size(map_list)
        else:
            map_index = (episode_index % (TOTAL_SCENARIO * REPLAY_TIMES)) // REPLAY_TIMES + 1
        
        self.map_path = map_dir + '/' + map_list[map_index]
        # ex专用
        # self.map_path = map_dir + '/' + "2343.png"
        print(f'Loading map: {self.map_path}')
        # 加载地图文件并转换为整数类型
        ground_truth = (io.imread(self.map_path, 1)).astype(int)

        # 对地图进行下采样，降低分辨率，每 2x2 像素块取最小值
        ground_truth = block_reduce(ground_truth, 2, np.min)
        
        # 提取机器人初始位置，找到值为 208 的像素点，并选择第 10 个候选点
        robot_cell = np.array(np.nonzero(ground_truth == 208))
        robot_cell = np.array([robot_cell[1, 10], robot_cell[0, 10]])

        # 将地图二值化，标记可通行区域和不可通行区域
        ground_truth = (ground_truth > 150) | ((ground_truth <= 80) & (ground_truth >= 50))
        ground_truth = ground_truth * 254 + 1  # 可通行区域为 255，不可通行区域为 1

        # 返回处理后的地图和机器人初始位置
        return ground_truth, robot_cell

    def update_robot_location(self, robot_location):
        # 更新机器人的当前位置。
        self.robot_location = robot_location
        self.robot_cell = np.array([round((robot_location[0] - self.belief_origin_x) / self.cell_size),
                                    round((robot_location[1] - self.belief_origin_y) / self.cell_size)])

    def update_robot_belief(self, robot_cell):
        # 更新机器人对环境的信念地图。
        self.robot_belief = sensor_work(robot_cell, round(self.sensor_range / self.cell_size), self.robot_belief,
                                        self.ground_truth)

    def calculate_reward(self):
        # 计算当前探索状态的奖励值。
        reward = 0

        global_frontiers = get_frontier_in_map(self.belief_info)
        if global_frontiers.shape[0] == 0:
            delta_num = self.global_frontiers.shape[0]
        else:
            global_frontiers = global_frontiers.reshape(-1, 2)
            frontiers_to_check = global_frontiers[:, 0] + global_frontiers[:, 1] * 1j
            pre_frontiers_to_check = self.global_frontiers[:, 0] + self.global_frontiers[:, 1] * 1j
            frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
            pre_frontiers_num = pre_frontiers_to_check.shape[0]
            delta_num = pre_frontiers_num - frontiers_num

        reward += delta_num / 40

        self.global_frontiers = global_frontiers
        self.old_belief = deepcopy(self.robot_belief)

        return reward

    def check_done(self):
        # 检查任务是否完成。
        if np.sum(self.ground_truth == 255) - np.sum(self.robot_belief == 255) <= 250:
            self.done = True

    def evaluate_exploration_rate(self):
        # 评估当前的探索进度。
        self.explored_rate = np.sum(self.robot_belief == 255) / np.sum(self.ground_truth == 255)

    def step(self, next_waypoint, agent_id):
        # 执行一步探索操作，更新信念地图和机器人位置。
        self.evaluate_exploration_rate()
        self.robot_locations[agent_id] = next_waypoint
        reward = 0
        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)
        self.update_robot_belief(cell)
        return reward
    
    def update_robot_location(self, next_waypoint, agent_id):
        self.robot_locations[agent_id] = next_waypoint


