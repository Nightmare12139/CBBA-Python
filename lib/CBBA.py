#!/usr/bin/env python3
import sys
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
from Task import Task
from WorldInfo import WorldInfo


class CBBA(object):
    num_agents: int  # 智能体数量
    num_tasks: int  # 任务数量
    max_depth: int  # 任务包最大深度
    time_window_flag: bool  # 若存在时间窗则为 True
    duration_flag: bool  # 当所有任务持续时间都大于 0 时为 True
    agent_types: list
    task_types: list
    space_limit_x: list  # x 坐标范围 [min, max]，单位米
    space_limit_y: list  # y 坐标范围 [min, max]，单位米
    space_limit_z: list  # z 坐标范围 [min, max]，单位米
    time_interval_list: list  # 所有智能体与任务的时间区间
    agent_index_list: list  # 一维列表
    bundle_list: list  # 二维列表
    path_list: list  # 二维列表
    times_list: list  # 二维列表
    scores_list: list  # 二维列表
    bid_list: list  # 二维列表
    winners_list: list  # 二维列表
    winner_bid_list: list  # 二维列表
    graph: list  # 二维列表，表示图结构
    AgentList: list  # 一维列表，每个元素是 Agent 数据类
    TaskList: list  # 一维列表，每个元素是 Task 数据类
    WorldInfo: WorldInfo  # WorldInfo 数据类对象

    def __init__(self, config_data: dict):
        """
        构造函数
        初始化 CBBA 参数

        config_data: dict，包含所有配置
            config_file_name = "config.json"
            json_file = open(config_file_name)
            config_data = json.load(json_file)
        """

        # 智能体类型列表
        self.agent_types = config_data["AGENT_TYPES"]
        # 任务类型列表
        self.task_types = config_data["TASK_TYPES"]
        # 所有智能体与任务的时间区间
        self.time_interval_list = [min(int(config_data["TRACK_DEFAULT"]["START_TIME"]),
                                       int(config_data["RESCUE_DEFAULT"]["START_TIME"])),
                                   max(int(config_data["TRACK_DEFAULT"]["END_TIME"]),
                                       int(config_data["RESCUE_DEFAULT"]["END_TIME"]))]
        # 当所有任务持续时间都大于 0 时为 True
        self.duration_flag = (min(int(config_data["TRACK_DEFAULT"]["DURATION"]),
                                  int(config_data["RESCUE_DEFAULT"]["DURATION"])) > 0)

        # 初始化兼容性矩阵
        self.compatibility_mat = [[0] * len(self.task_types) for _ in range(len(self.agent_types))]

        # 供用户扩展：设置智能体-任务类型匹配关系（哪些智能体类型可执行哪些任务类型）
        try:
            # 四旋翼可执行 track 任务
            self.compatibility_mat[self.agent_types.index("quad")][self.task_types.index("track")] = 1
        except Exception as e:
            print(e)

        try:
            # 车辆可执行 rescue 任务
            self.compatibility_mat[self.agent_types.index("car")][self.task_types.index("rescue")] = 1
        except Exception as e:
            print(e)

    def settings(self, AgentList: list, TaskList: list, WorldInfoInput: WorldInfo,
                 max_depth: int, time_window_flag: bool):
        """
        基于新的 AgentList、TaskList 和 WorldInfoInput 初始化相关列表。
        """

        self.num_agents = len(AgentList)
        self.num_tasks = len(TaskList)
        self.max_depth = max_depth
        self.time_window_flag = time_window_flag

        self.AgentList = AgentList
        self.TaskList = TaskList

        # 世界信息
        self.WorldInfo = WorldInfoInput
        self.space_limit_x = self.WorldInfo.limit_x
        self.space_limit_y = self.WorldInfo.limit_y
        self.space_limit_z = self.WorldInfo.limit_z

        # 全连接图
        # 二维列表
        self.graph = np.logical_not(np.identity(self.num_agents)).tolist()

        # 初始化这些属性
        self.bundle_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.path_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.times_list = [[-1] * self.max_depth for _ in range(self.num_agents)]
        self.scores_list = [[-1] * self.max_depth for _ in range(self.num_agents)]

        # 修正初始化：由 0 向量改为 -1 向量
        self.bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.winners_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]
        self.winner_bid_list = [[-1] * self.num_tasks for _ in range(self.num_agents)]

        self.agent_index_list = []
        for n in range(self.num_agents):
            self.agent_index_list.append(self.AgentList[n].agent_id)

    def solve(self, AgentList: list, TaskList: list, WorldInfoInput: WorldInfo,
              max_depth: int, time_window_flag: bool):
        """
        CBBA 主函数
        """

        # 根据 AgentList、TaskList 和 WorldInfoInput 初始化相关列表。
        self.settings(AgentList, TaskList, WorldInfoInput, max_depth, time_window_flag)

        # 初始化工作变量
        # 当前迭代次数
        iter_idx = 1
        # 当前赢家信息更新时间矩阵
        time_mat = [[0] * self.num_agents for _ in range(self.num_agents)]
        iter_prev = 0
        done_flag = False

        # CBBA 主循环（直到收敛）
        while not done_flag:

            # 1. 通信阶段
            # 对赢家和出价执行一致性同步
            time_mat = self.communicate(time_mat, iter_idx)

            # 2. 执行 CBBA 任务包构建/更新
            # 在每个智能体上运行 CBBA（分布式但同步）
            for idx_agent in range(self.num_agents):
                new_bid_flag = self.bundle(idx_agent)

                # 更新最近一次发生变化的迭代
                # 用于收敛判定，最终实现中可移除
                if new_bid_flag:
                    iter_prev = iter_idx

            # 3. 收敛检查
            # 判断分配是否结束（当前版本启用，后续可改为持续循环）
            if (iter_idx - iter_prev) > self.num_agents:
                done_flag = True
            elif (iter_idx - iter_prev) > (2*self.num_agents):
                print("Algorithm did not converge due to communication trouble")
                done_flag = True
            else:
                # 保持循环继续
                iter_idx += 1

        # 将路径与任务包中的索引映射为真实任务 ID
        for n in range(self.num_agents):
            for m in range(self.max_depth):
                if self.bundle_list[n][m] == -1:
                    break
                else:
                    self.bundle_list[n][m] = self.TaskList[self.bundle_list[n][m]].task_id

                if self.path_list[n][m] == -1:
                    break
                else:
                    self.path_list[n][m] = self.TaskList[self.path_list[n][m]].task_id

        # 计算 CBBA 分配总得分
        score_total = 0
        for n in range(self.num_agents):
            for m in range(self.max_depth):
                if self.scores_list[n][m] > -1:
                    score_total += self.scores_list[n][m]
                else:
                    break

        # 输出每个智能体的结果路径，移除所有 -1
        self.path_list = [list(filter(lambda a: a != -1, self.path_list[i]))
                          for i in range(len(self.path_list))]

        # 删除冗余元素
        self.bundle_list = [list(filter(lambda a: a != -1, self.bundle_list[i]))
                            for i in range(len(self.bundle_list))]

        self.times_list = [list(filter(lambda a: a != -1, self.times_list[i]))
                           for i in range(len(self.times_list))]

        self.scores_list = [list(filter(lambda a: a != -1, self.scores_list[i]))
                            for i in range(len(self.scores_list))]

        return self.path_list, self.times_list

    def bundle(self, idx_agent: int):
        """
        CBBA 的包构建/更新主流程（在每个智能体上执行）
        """

        # 通信后更新任务包，移除被超价任务
        self.bundle_remove(idx_agent)
        # 对新任务出价并加入任务包
        new_bid_flag = self.bundle_add(idx_agent)

        return new_bid_flag

    def bundle_remove(self, idx_agent: int):
        """
        通信后更新任务包
        对于被超价的智能体，释放其任务包中的任务
        """

        out_bid_for_task = False
        for idx in range(self.max_depth):
            # 若 bundle(j) < 0，表示到 j 之前的任务均有效
            # 且仍在路径中，其余（j 到 MAX_DEPTH）将被释放
            if self.bundle_list[idx_agent][idx] < 0:
                break
            else:
                # 检查该任务是否被他人超价；若是，则释放该任务及其后续路径任务
                if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] != self.agent_index_list[idx_agent]:
                    out_bid_for_task = True

                if out_bid_for_task:
                    # 若先前任务已丢失，则当前任务也需释放
                    if self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] == \
                            self.agent_index_list[idx_agent]:
                        # 若仍在赢家列表中则移除
                        self.winners_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1
                        self.winner_bid_list[idx_agent][self.bundle_list[idx_agent][idx]] = -1

                    # 从路径和时间向量中清除，并从任务包中移除
                    path_current = copy.deepcopy(self.path_list[idx_agent])
                    idx_remove = path_current.index(self.bundle_list[idx_agent][idx])

                    # 删除 idx_remove 位置元素，并在末尾补 -1
                    del self.path_list[idx_agent][idx_remove]
                    self.path_list[idx_agent].append(-1)
                    del self.times_list[idx_agent][idx_remove]
                    self.times_list[idx_agent].append(-1)
                    del self.scores_list[idx_agent][idx_remove]
                    self.scores_list[idx_agent].append(-1)

                    self.bundle_list[idx_agent][idx] = -1

    def bundle_add(self, idx_agent: int):
        """
        为每个智能体创建任务包
        """

        epsilon = 1e-5
        new_bid_flag = False

        # 检查任务包是否已满（bundle_full_flag 为 True 表示已满）
        index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
        if len(index_array) > 0:
            bundle_full_flag = False
        else:
            bundle_full_flag = True
        
        # 初始化可行性矩阵（记录哪些插入位置 j 可剪枝）
        # feasibility = np.ones((self.num_tasks, self.max_depth+1))
        feasibility = [[1] * (self.max_depth+1) for _ in range(self.num_tasks)]

        while not bundle_full_flag:
            # 基于当前分配更新任务价值
            [best_indices, task_times, feasibility] = self.compute_bid(idx_agent, feasibility)

            # 判断哪些分配可用。array_logical_1、array_logical_2、
            # array_logical_3 均为 numpy 一维布尔数组
            array_logical_1 = ((np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent]))
                               > epsilon)
            # 找出相等项
            array_logical_2 = (abs(np.array(self.bid_list[idx_agent]) - np.array(self.winner_bid_list[idx_agent]))
                               <= epsilon)
            # 基于智能体索引进行平局判定
            array_logical_3 = (self.agent_index_list[idx_agent] < np.array(self.winners_list[idx_agent]))

            array_logical_result = np.logical_or(array_logical_1, np.logical_and(array_logical_2, array_logical_3))

            # 选择提升得分最大的分配并出价
            array_max = np.array(self.bid_list[idx_agent]) * array_logical_result
            best_task = array_max.argmax()
            value_max = max(array_max)

            if value_max > 0:
                # 设置新出价标志
                new_bid_flag = True

                # 检查平局情况，返回一维 numpy 数组
                all_values = np.where(array_max == value_max)[0]
                if len(all_values) == 1:
                    best_task = all_values[0]
                else:
                    # 以任务开始时间更早者打破平局
                    earliest = sys.float_info.max
                    for i in range(len(all_values)):
                        if self.TaskList[all_values[i]].start_time < earliest:
                            earliest = self.TaskList[all_values[i]].start_time
                            best_task = all_values[i]
                
                self.winners_list[idx_agent][best_task] = self.AgentList[idx_agent].agent_id
                self.winner_bid_list[idx_agent][best_task] = self.bid_list[idx_agent][best_task]

                # 在指定索引插入元素，并删除原列表最后一个元素
                self.path_list[idx_agent].insert(best_indices[best_task], best_task)
                del self.path_list[idx_agent][-1]
                self.times_list[idx_agent].insert(best_indices[best_task], task_times[best_task])
                del self.times_list[idx_agent][-1]
                self.scores_list[idx_agent].insert(best_indices[best_task], self.bid_list[idx_agent][best_task])
                del self.scores_list[idx_agent][-1]

                length = len(np.where(np.array(self.bundle_list[idx_agent]) > -1)[0])
                self.bundle_list[idx_agent][length] = best_task

                # 更新可行性
                # 将相同可行性布尔值插入可行性矩阵
                for i in range(self.num_tasks):
                    # 在指定索引插入元素，并删除原列表最后一个元素
                    feasibility[i].insert(best_indices[best_task], feasibility[i][best_indices[best_task]])
                    del feasibility[i][-1]
            else:
                break

            # 检查任务包是否已满
            index_array = np.where(np.array(self.bundle_list[idx_agent]) == -1)[0]
            if len(index_array) > 0:
                bundle_full_flag = False
            else:
                bundle_full_flag = True

        return new_bid_flag

    def communicate(self, time_mat: list, iter_idx: int):
        """
        在邻居之间执行一致性通信，检查并解决各智能体之间的冲突。
        这里实现的是如下论文表 1 中的消息传递机制：
        "Consensus-Based Decentralized Auctions for Robust Task Allocation",
        H.-L. Choi, L. Brunet, and J. P. How, IEEE Transactions on Robotics,
        Vol. 25, (4): 912 - 926, August 2009

        注：表 1 描述的是智能体 i 在接收智能体 k 关于任务 j 的信息后应采取的动作规则。
        下面包含大量 if-else 的大循环就是对表 1 的直接实现，便于对照阅读。
        """

        # time_mat 为当前赢家信息更新时间矩阵
        # iter_idx 为当前迭代次数

        time_mat_new = copy.deepcopy(time_mat)

        # 复制数据
        old_z = copy.deepcopy(self.winners_list)
        old_y = copy.deepcopy(self.winner_bid_list)
        z = copy.deepcopy(old_z)
        y = copy.deepcopy(old_y)

        epsilon = 10e-6

        # 开始智能体间通信
        # 发送方 = k
        # 接收方 = i
        # 任务 = j

        for k in range(self.num_agents):
            for i in range(self.num_agents):
                if self.graph[k][i] == 1:
                    for j in range(self.num_tasks):
                        # 按每个任务实现规则表

                        # 条目 1~4：发送方认为任务归自己
                        if old_z[k][j] == k:
                            
                            # 条目 1：更新或保持
                            if z[i][j] == i:
                                if (old_y[k][j] - y[i][j]) > epsilon:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # 分数相等
                                    if z[i][j] > old_z[k][j]:  # 以更小索引打破平局
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # 条目 2：更新
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]
                    
                            # 条目 3：更新或保持
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif (old_y[k][j] - y[i][j]) > epsilon:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # 分数相等
                                    if z[i][j] > old_z[k][j]:  # 以更小索引打破平局
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]

                            # 条目 4：更新
                            elif z[i][j] == -1:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # 条目 5~8：发送方认为接收方拥有该任务
                        elif old_z[k][j] == i:

                            # 条目 5：保持
                            if z[i][j] == i:
                                # 不执行任何操作
                                pass
                                
                            # 条目 6：重置
                            elif z[i][j] == k:
                                z[i][j] = -1
                                y[i][j] = -1

                            # 条目 7：重置或保持
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # 重置
                                    z[i][j] = -1
                                    y[i][j] = -1
                                
                            # 条目 8：保持
                            elif z[i][j] == -1:
                                # 不执行任何操作
                                pass

                            else:
                                print(z[i][j])
                                raise Exception("Unknown winner value: please revise!")

                        # 条目 9~13：发送方认为其他智能体拥有该任务
                        elif old_z[k][j] > -1:
                            
                            # 条目 9：更新或保持
                            if z[i][j] == i:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                    if (old_y[k][j] - y[i][j]) > epsilon:
                                        z[i][j] = old_z[k][j]  # 更新
                                        y[i][j] = old_y[k][j]
                                    elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # 分数相等
                                        if z[i][j] > old_z[k][j]:  # 以更小索引打破平局
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]

                            # 条目 10：更新或重置
                            elif z[i][j] == k:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]
                                else:  # 重置
                                    z[i][j] = -1
                                    y[i][j] = -1

                            # 条目 11：更新或保持
                            elif z[i][j] == old_z[k][j]:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # 条目 12：更新、重置或保持
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:
                                    if time_mat[k][old_z[k][j]] >= time_mat_new[i][old_z[k][j]]:  # 更新
                                        z[i][j] = old_z[k][j]
                                        y[i][j] = old_y[k][j]
                                    elif time_mat[k][old_z[k][j]] < time_mat_new[i][old_z[k][j]]:  # 重置
                                        z[i][j] = -1
                                        y[i][j] = -1
                                    else:
                                        raise Exception("Unknown condition for Entry 12: please revise!")
                                else:
                                    if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:
                                        if (old_y[k][j] - y[i][j]) > epsilon:  # 更新
                                            z[i][j] = old_z[k][j]
                                            y[i][j] = old_y[k][j]
                                        elif abs(old_y[k][j] - y[i][j]) <= epsilon:  # 分数相等
                                            if z[i][j] > old_z[k][j]:  # 以更小索引打破平局
                                                z[i][j] = old_z[k][j]
                                                y[i][j] = old_y[k][j]

                            # 条目 13：更新或保持
                            elif z[i][j] == -1:
                                if time_mat[k][old_z[k][j]] > time_mat_new[i][old_z[k][j]]:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            else:
                                raise Exception("Unknown winner value: please revise!")

                        # 条目 14~17：发送方认为无人拥有该任务
                        elif old_z[k][j] == -1:

                            # 条目 14：保持
                            if z[i][j] == i:
                                # 不执行任何操作
                                pass

                            # 条目 15：更新
                            elif z[i][j] == k:
                                z[i][j] = old_z[k][j]
                                y[i][j] = old_y[k][j]

                            # 条目 16：更新或保持
                            elif z[i][j] > -1:
                                if time_mat[k][z[i][j]] > time_mat_new[i][z[i][j]]:  # 更新
                                    z[i][j] = old_z[k][j]
                                    y[i][j] = old_y[k][j]

                            # 条目 17：保持
                            elif z[i][j] == -1:
                                # 不执行任何操作
                                pass
                            else:
                                raise Exception("Unknown winner value: please revise!")

                            # 规则表结束
                        else:
                            raise Exception("Unknown winner value: please revise!")

                    # 基于最新通信结果更新所有智能体时间戳
                    for n in range(self.num_agents):
                        if (n != i) and (time_mat_new[i][n] < time_mat[k][n]):
                            time_mat_new[i][n] = time_mat[k][n]
                    time_mat_new[i][k] = iter_idx

        # 复制数据
        self.winners_list = copy.deepcopy(z)
        self.winner_bid_list = copy.deepcopy(y)
        return time_mat_new

    def compute_bid(self, idx_agent: int, feasibility: list):
        """
        计算每个任务的出价，返回出价值、任务在路径中的最优插入位置以及新路径对应时间。
        """

        # 若路径已满，则无法再添加任务
        empty_task_index_list = np.where(np.array(self.path_list[idx_agent]) == -1)[0]
        if len(empty_task_index_list) == 0:
            best_indices = []
            task_times = []
            feasibility = []
            return best_indices, task_times, feasibility

        # 重置出价、路径最优位置和最优时间
        self.bid_list[idx_agent] = [-1] * self.num_tasks
        best_indices = [-1] * self.num_tasks
        task_times = [-2] * self.num_tasks

        # 对每个任务遍历
        for idx_task in range(self.num_tasks):
            # 检查智能体与任务的兼容性
            # 浮点精度冗余判断
            if self.compatibility_mat[self.AgentList[idx_agent].agent_type][self.TaskList[idx_task].task_type] > 0.5:
                
                # 检查路径中是否已包含任务 m
                index_array = np.where(np.array(self.path_list[idx_agent][0:empty_task_index_list[0]]) == idx_task)[0]
                if len(index_array) < 0.5:
                    # 该任务尚未在当前任务包中
                    # 通过插入到当前路径中寻找可获得的最佳得分
                    best_bid = 0
                    best_index = -1
                    best_time = -2

                    # 尝试将任务 m 插入位置 j，检查是否能得到更优新路径
                    for j in range(empty_task_index_list[0]+1):
                        if feasibility[idx_task][j] == 1:
                            # 检查新路径可行性：True 表示跳过本次，False 表示可行
                            skip_flag = False

                            if j == 0:
                                # 插入到开头
                                task_prev = []
                                time_prev = []
                            else:
                                Task_temp = self.TaskList[self.path_list[idx_agent][j-1]]
                                task_prev = Task(**Task_temp.__dict__)
                                time_prev = self.times_list[idx_agent][j-1]
                            
                            if j == (empty_task_index_list[0]):
                                task_next = []
                                time_next = []
                            else:
                                Task_temp = self.TaskList[self.path_list[idx_agent][j]]
                                task_next = Task(**Task_temp.__dict__)
                                time_next = self.times_list[idx_agent][j]

                            # 计算最早/最晚开始时间与得分
                            Task_temp = self.TaskList[idx_task]
                            [score, min_start, max_start] = self.scoring_compute_score(
                                idx_agent, Task(**Task_temp.__dict__), task_prev, time_prev, task_next, time_next)

                            if self.time_window_flag:
                                # 若任务具有时间窗
                                if min_start > max_start:
                                    # 路径不可行
                                    skip_flag = True
                                    feasibility[idx_task][j] = 0

                                if not skip_flag:
                                    # 保存最佳得分和任务位置
                                    if score > best_bid:
                                        best_bid = score
                                        best_index = j
                                        # 选择最早开始时间作为最优
                                        best_time = min_start
                            else:
                                # 任务无时间窗
                                # 保存最佳得分和任务位置
                                if score > best_bid:
                                    best_bid = score
                                    best_index = j
                                    # 选择最早开始时间作为最优
                                    best_time = 0.0

                    # 保存最佳出价信息
                    if best_bid > 0:
                        self.bid_list[idx_agent][idx_task] = best_bid
                        best_indices[idx_task] = best_index
                        task_times[idx_task] = best_time

            # 该任务与当前智能体类型不兼容
        # 任务遍历结束
        return best_indices, task_times, feasibility

    def scoring_compute_score(self, idx_agent: int, task_current: Task, task_prev: Task,
                              time_prev, task_next: Task, time_next):
        """
        计算执行任务的边际得分，并返回该任务的预计开始时间。
        """

        if (self.AgentList[idx_agent].agent_type == self.agent_types.index("quad")) or \
                (self.AgentList[idx_agent].agent_type == self.agent_types.index("car")):
            
            if not task_prev:
                # 路径中的第一个任务
                # 计算任务开始时间
                dt = math.sqrt((self.AgentList[idx_agent].x-task_current.x)**2 +
                               (self.AgentList[idx_agent].y-task_current.y)**2 +
                               (self.AgentList[idx_agent].z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                min_start = max(task_current.start_time, self.AgentList[idx_agent].availability + dt)
            else:
                # 非路径首任务
                dt = math.sqrt((task_prev.x-task_current.x)**2 + (task_prev.y-task_current.y)**2 +
                               (task_prev.z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                # 必须预留完成 j-1 任务并前往任务 m 的时间
                min_start = max(task_current.start_time, time_prev + task_prev.duration + dt)

            if not task_next:
                # 路径中的最后一个任务
                dt = 0.0
                max_start = task_current.end_time
            else:
                # 非末任务，检查是否仍可完成后续承诺任务
                dt = math.sqrt((task_next.x-task_current.x)**2 + (task_next.y-task_current.y)**2 +
                               (task_next.z-task_current.z)**2) / self.AgentList[idx_agent].nom_velocity
                # 必须预留完成任务 m 并飞往 j+1 任务的时间
                max_start = min(task_current.end_time, time_next - task_current.duration - dt)

            # 计算得分
            if self.time_window_flag:
                # 若任务具有时间窗
                reward = task_current.task_value * \
                         math.exp((-task_current.discount) * (min_start-task_current.start_time))
            else:
                # 任务无时间窗
                dt_current = math.sqrt((self.AgentList[idx_agent].x-task_current.x)**2 +
                                       (self.AgentList[idx_agent].y-task_current.y)**2 +
                                       (self.AgentList[idx_agent].z-task_current.z)**2) / \
                             self.AgentList[idx_agent].nom_velocity

                reward = task_current.task_value * math.exp((-task_current.discount) * dt_current)

            # # 扣除油耗成本。可采用常量油耗以满足 DMG（边际收益递减）。
            # # 该分数是近似值，因为油耗被重复计算；不应用于与最优解直接比较。
            # # 需要在 CBBA 算法完成后再计算路径真实得分。
            # penalty = self.AgentList[idx_agent].fuel * math.sqrt(
            #     (self.AgentList[idx_agent].x-task_current.x)**2 + (self.AgentList[idx_agent].y-task_current.y)**2 +
            #     (self.AgentList[idx_agent].z-task_current.z)**2)
            #
            # score = reward - penalty

            score = reward
        else:
            # 供用户扩展：为特定智能体定义评分函数，例如：
            # elseif(agent.type == CBBA_Params.AGENT_TYPES.NEW_AGENT), ...
            # 需要定义 score、minStart 和 maxStart
            raise Exception("Unknown agent type!")

        return score, min_start, max_start

    def plot_assignment(self):
        """
        绘制带任务时间窗时的 CBBA 输出结果。
        """

        # 3D 绘图
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        # 3D 空间中文本偏移量
        offset = (self.WorldInfo.limit_x[1]-self.WorldInfo.limit_x[0]) / 50

        # 绘制任务
        for m in range(self.num_tasks):
            # 追踪类任务显示为红色
            if self.TaskList[m].task_type == 0:
                color_str = 'red'
            # 救援类任务显示为蓝色
            else:
                color_str = 'blue'

            ax_3d.scatter([self.TaskList[m].x]*2, [self.TaskList[m].y]*2,
                          [self.TaskList[m].start_time, self.TaskList[m].end_time], marker='x', color=color_str)
            ax_3d.plot3D([self.TaskList[m].x]*2, [self.TaskList[m].y]*2,
                         [self.TaskList[m].start_time, self.TaskList[m].end_time],
                         linestyle=':', color=color_str, linewidth=3)
            ax_3d.text(self.TaskList[m].x+offset, self.TaskList[m].y+offset, self.TaskList[m].start_time, "T"+str(m))

        # 绘制智能体
        for n in range(self.num_agents):
            # 四旋翼智能体显示为红色
            if self.AgentList[n].agent_type == 0:
                color_str = 'red'
            # 车辆智能体显示为蓝色
            else:
                color_str = 'blue'
            ax_3d.scatter(self.AgentList[n].x, self.AgentList[n].y, 0, marker='o', color=color_str)
            ax_3d.text(self.AgentList[n].x+offset, self.AgentList[n].y+offset, 0.1, "A"+str(n))

            # 若路径非空
            if self.path_list[n]:
                Task_prev = self.lookup_task(self.path_list[n][0])
                ax_3d.plot3D([self.AgentList[n].x, Task_prev.x], [self.AgentList[n].y, Task_prev.y],
                             [0, self.times_list[n][0]], linewidth=2, color=color_str)
                ax_3d.plot3D([Task_prev.x, Task_prev.x], [Task_prev.y, Task_prev.y],
                             [self.times_list[n][0], self.times_list[n][0]+Task_prev.duration],
                             linewidth=2, color=color_str)

                for m in range(1, len(self.path_list[n])):
                    if self.path_list[n][m] > -1:
                        Task_next = self.lookup_task(self.path_list[n][m])
                        ax_3d.plot3D([Task_prev.x, Task_next.x], [Task_prev.y, Task_next.y],
                                     [self.times_list[n][m-1]+Task_prev.duration, self.times_list[n][m]],
                                     linewidth=2, color=color_str)
                        ax_3d.plot3D([Task_next.x, Task_next.x], [Task_next.y, Task_next.y],
                                     [self.times_list[n][m], self.times_list[n][m]+Task_next.duration],
                                     linewidth=2, color=color_str)
                        Task_prev = Task(**Task_next.__dict__)
        
        plt.title('Agent Paths with Time Windows')
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Time")
        ax_3d.set_aspect('auto')

        # 设置图例
        colors = ["red", "blue", "red", "blue"]
        marker_list = ["o", "o", "x", "x"]
        labels = ["Agent type 1", "Agent type 2", "Task type 1", "Task type 2"]
        def f(marker_type, color_type): return plt.plot([], [], marker=marker_type, color=color_type, ls="none")[0]
        handles = [f(marker_list[i], colors[i]) for i in range(len(labels))]
        plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)
        self.set_axes_equal_xy(ax_3d, flag_3d=True)

        if self.duration_flag:
            # 绘制智能体时间安排
            fig_schedule = plt.figure(2)
            fig_schedule.suptitle("Schedules for Agents")
            for idx_agent in range(self.num_agents):
                ax = plt.subplot(self.num_agents, 1, idx_agent+1)
                ax.set_title("Agent "+str(idx_agent))
                if idx_agent == (self.num_agents - 1):
                    ax.set_xlabel("Time [sec]")
                ax.set_xlim(self.time_interval_list)
                ax.set_ylim([0.95, 1.05])

                # 四旋翼智能体显示为红色
                if self.AgentList[idx_agent].agent_type == 0:
                    color_str = 'red'
                # 车辆智能体显示为蓝色
                else:
                    color_str = 'blue'

                if self.path_list[idx_agent]:
                    for idx_path in range(len(self.path_list[idx_agent])):
                        if self.path_list[idx_agent][idx_path] > -1:
                            task_current = self.lookup_task(self.path_list[idx_agent][idx_path])
                            ax.plot([self.times_list[idx_agent][idx_path],
                                     self.times_list[idx_agent][idx_path]+task_current.duration], [1, 1],
                                    linestyle='-', linewidth=10, color=color_str, alpha=0.5)
                            ax.plot([task_current.start_time, task_current.end_time], [1, 1], linestyle='-.',
                                    linewidth=2, color=color_str)

            # 设置图例
            colors = ["red", "red"]
            line_styles = ["-", "-."]
            line_width_list = [10, 2]
            labels = ["Assignment Time", "Task Time"]
            def f(line_style, color_type, line_width): return plt.plot([], [], linestyle=line_style, color=color_type,
                                                                       linewidth=line_width)[0]
            handles = [f(line_styles[i], colors[i], line_width_list[i]) for i in range(len(labels))]
            fig_schedule.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)

        plt.show(block=False)

    def plot_assignment_without_timewindow(self):
        """
        绘制无任务时间窗时的 CBBA 输出结果。
        """

        # 3D 绘图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # 3D 空间中文本偏移量
        offset = (self.WorldInfo.limit_x[1]-self.WorldInfo.limit_x[0]) / 100

        # 绘制任务
        for m in range(self.num_tasks):
            # 追踪类任务显示为红色
            if self.TaskList[m].task_type == 0:
                color_str = 'red'
            # 救援类任务显示为蓝色
            else:
                color_str = 'blue'
            ax.scatter(self.TaskList[m].x, self.TaskList[m].y, marker='x', color=color_str)
            ax.text(self.TaskList[m].x+offset, self.TaskList[m].y+offset, "T"+str(m))

        # 绘制智能体
        for n in range(self.num_agents):
            # 四旋翼智能体显示为红色
            if self.AgentList[n].agent_type == 0:
                color_str = 'red'
            # 车辆智能体显示为蓝色
            else:
                color_str = 'blue'
            ax.scatter(self.AgentList[n].x, self.AgentList[n].y, marker='o', color=color_str)
            ax.text(self.AgentList[n].x+offset, self.AgentList[n].y+offset, "A"+str(n))

            # 若路径非空
            if self.path_list[n]:
                Task_prev = self.lookup_task(self.path_list[n][0])
                ax.plot([self.AgentList[n].x, Task_prev.x], [self.AgentList[n].y, Task_prev.y],
                        linewidth=2, color=color_str)
                for m in range(1, len(self.path_list[n])):
                    if self.path_list[n][m] > -1:
                        Task_next = self.lookup_task(self.path_list[n][m])
                        ax.plot([Task_prev.x, Task_next.x], [Task_prev.y, Task_next.y], linewidth=2, color=color_str)
                        Task_prev = Task(**Task_next.__dict__)
        
        plt.title('Agent Paths without Time Windows')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # 设置图例
        colors = ["red", "blue", "red", "blue"]
        marker_list = ["o", "o", "x", "x"]
        labels = ["Agent type 1", "Agent type 2", "Task type 1", "Task type 2"]
        def f(marker_type, color_type): return plt.plot([], [], marker=marker_type, color=color_type, ls="none")[0]
        handles = [f(marker_list[i], colors[i]) for i in range(len(labels))]
        plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)

        self.set_axes_equal_xy(ax, flag_3d=False)
        plt.show(block=False)

    def set_axes_equal_xy(self, ax, flag_3d: bool):
        """
        仅让 3D 图中的 x、y 轴保持等比例。用于解决 Matplotlib 中
        ax.set_aspect('equal') 与 ax.axis('equal') 在 3D 场景下不生效的问题。
        参考：https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

        输入
        ax: matplotlib 坐标轴对象，例如 plt.gca() 的输出。
        flag_3d: 布尔值，若为 True 表示 3D 图。
        """

        x_limits = self.space_limit_x
        y_limits = self.space_limit_y

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)

        # 从无穷范数意义上看，绘图包围盒可视作球体
        # 因此将最大跨度的一半定义为绘图半径。
        plot_radius = 0.5*max([x_range, y_range])

        if flag_3d:
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            if abs(self.time_interval_list[1]) >= 1e-3:
                ax.set_zlim3d(self.time_interval_list)
        else:
            ax.set_xlim([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim([y_middle - plot_radius, y_middle + plot_radius])

    def lookup_task(self, task_id: int):
        """
        根据任务 ID 查找对应 Task。
        """

        TaskOutput = []
        for m in range(self.num_tasks):
            if self.TaskList[m].task_id == task_id:
                Task_temp = self.TaskList[m]
                TaskOutput.append(Task(**Task_temp.__dict__))

        if not TaskOutput:
            raise Exception("Task " + str(task_id) + " not found!")

        return TaskOutput[0]
