import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import RMSprop

from players.base import PlayerBase
from models.nn import DQN

class DQNPlayer(PlayerBase):
    def __init__(self, 
                 strageties : list ,
                 observation_size : int,
                 device : torch.device,
                 lr=.01,
                 batch_size=32,
                 pos_action_prob=.95,
                 reward_decay_ratio=.95,
                 max_experience_size=1e5,
                 update_experience_interval=5,
                 update_model_interval=10,
                 initial_score=1., 
                 mul_ratio=1.,
                 *args, 
                 **kwargs
                ):
        """初始化深度Q网络

        Args:
            observation_size (int): 博弈参与者可观测数据，如其他博弈者的选择.
            device (torch.device): 模型运行设备.
            lr (float, optional): 学习率. 默认值为 0.01.
            batch_size (int, optional): 批量梯度下降数量. 默认值为 32.
            pos_action_prob (float, optional): 跟随梯度随机进行参数梯度下降概率. 默认值为 0.95.
            reward_decay_ratio (float, optional): 下一次目标价值衰减率. 默认值为 0.95.
            max_experience_size (_type_, optional): 经验样本上限数量. 默认值为 1e5.
            update_experience_interval (int, optional): 经验数据集更新间隔. 默认值为 5.
            update_model_interval (int, optional): 模型权重更新间隔. 默认值为 10.
            initial_score (float) : 初始分数
            mul_ratio (float) : 博弈者得分/罚分接受比率系数
            
        """
        super(DQNPlayer, self).__init__(strageties, initial_score, mul_ratio)
        self.player = DQN(observation_size,
                          len(strageties),
                          MSELoss,
                          RMSprop,
                          device,
                          lr,
                          batch_size,
                          pos_action_prob,
                          reward_decay_ratio,
                          max_experience_size,
                          update_experience_interval,
                          update_model_interval
                         )
        self.last_action = False
        
    def decide(self, state=None, *args, **kwargs):
        strategie_index = self.player.decide(state).numpy()
        strategie_name  = np.array(self.strategies)[strategie_index]
        return (strategie_index[0], strategie_name[0])
        
    def settle(self, val, *args, **kwargs):
        return self._reward(val) if val > 0 else self._punish(val)
        
    def learn(self, state, action, reward, next_state, *args, **kwargs):
        if self.last_action:
            self.player.update_experience(state, action, reward, next_state)
            self.player.learn()
        self.last_action = True
        
    def save_model(self, path, *args, **kwargs):
        self.player.save_model(path)
        
    def load_model(self, path, *args, **kwargs):
        self.load_model(path)