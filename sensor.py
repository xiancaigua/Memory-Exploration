import numpy as np


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    # 检查从起点到终点的路径是否存在碰撞，并更新信念地图。
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 2

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k != 1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    # 模拟传感器的工作过程，更新机器人对环境的信念地图。
    # 设置传感器的角度增量（以弧度为单位），每次扫描增加 0.5 度
    sensor_angle_inc = 0.5 / 180 * np.pi
    # 初始化传感器的起始角度为 0
    sensor_angle = 0
    # 获取机器人当前位置的 x 和 y 坐标
    x0 = robot_position[0]
    y0 = robot_position[1]
    
    # 循环遍历传感器的扫描角度，直到覆盖 360 度（2π 弧度）
    while sensor_angle < 2 * np.pi:
        # 根据当前角度计算传感器的终点坐标（x1, y1）
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        
        # 调用 collision_check 函数，更新机器人对环境的信念地图
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        
        # 增加传感器的扫描角度
        sensor_angle += sensor_angle_inc
    
    # 返回更新后的信念地图
    return robot_belief