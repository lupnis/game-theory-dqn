import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch.optim import Optimizer

from torch.nn.functional import one_hot
from collections import deque

class MLP(nn.Module):
    def __init__(self, input_size, hidden_list, activate_func=nn.Identity):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            nn.Linear(input_size, hidden_list[0])
        )
        for  (l_from, l_to) in zip(hidden_list[:-1], hidden_list[1:]):
            self.layers.append(activate_func())
            self.layers.append(nn.Linear(l_from, l_to))
    
    def forward(self, x):
        return self.layers(x)
       
    
class DQN(object):
    """深度Q网络
    
    """
    def __init__(self, 
                 observation_size : int,
                 strategy_count : int,
                 criterion : loss._Loss,
                 optimizer : Optimizer,
                 device=torch.device('cuda'),
                 lr=.01,
                 batch_size=32,
                 pos_action_prob=.95,
                 reward_decay_ratio=.95,
                 max_experience_size=1e5,
                 update_experience_interval=5,
                 update_model_interval=10,
                ):
        """初始化深度Q网络

        Args:
            observation_size (int): 博弈参与者可观测数据，如其他博弈者的选择.
            strategy_count (int): 博弈参与者可选策略总数.
            criterion (nn.modules.loss._Loss): 模型损失函数.
            optimizer (optim.Optimizer): 模型优化器.
            device (torch.device): 模型运行设备.
            lr (float, optional): 学习率. 默认值为 0.01.
            batch_size (int, optional): 批量梯度下降数量. 默认值为 32.
            pos_action_prob (float, optional): 跟随梯度随机进行参数梯度下降概率. 默认值为 0.95.
            reward_decay_ratio (float, optional): 下一次目标价值衰减率. 默认值为 0.95.
            max_experience_size (_type_, optional): 经验样本上限数量. 默认值为 1e5.
            update_experience_interval (int, optional): 经验数据集更新间隔. 默认值为 5.
            update_model_interval (int, optional): 模型权重更新间隔. 默认值为 10.
            
        """
        self.device                     = device
        self.observation_size           = observation_size
        self.strategy_count             = strategy_count
        self.eval_model                 = MLP(observation_size, [observation_size * 2, observation_size + 32, observation_size + 32, observation_size * 2, strategy_count], nn.LeakyReLU).to(device)
        self.target_model               = MLP(observation_size, [observation_size * 2, observation_size + 32, observation_size + 32, observation_size * 2, strategy_count], nn.LeakyReLU).to(device)
        self.criterion                  = criterion()
        self.optimizer                  = optimizer(self.eval_model.parameters(), lr=lr)
        self.batch_size                 = batch_size
        self.pos_action_prob            = pos_action_prob
        self.reward_decay_ratio         = reward_decay_ratio
        self.max_experience_size        = max_experience_size
        self.update_experience_interval = update_experience_interval
        self.update_model_interval      = update_model_interval
        
        self.experiences                = deque()

        self.cnt_step                   = 0
        self.cnt_experience             = 0
        
    def __call__(self, state):
        return self.decide(state)
    
    def decide(self, state):
        if state is None or torch.as_tensor(state).size()[0] == 0:
            return torch.randint(0, self.strategy_count, (1,))
        state = torch.as_tensor(state)
        state = state.to(self.device)
        state = torch.unsqueeze(state[-self.observation_size:], 0)
        rnd = torch.rand(1)[0]
        if float(rnd) >= self.pos_action_prob:
            return torch.randint(0, self.strategy_count, (1,))
        else:
            y_raw = self.eval_model(state.float())
            return torch.argmax(y_raw, 1).to('cpu').detach()
        
    def update_experience(self, state, action, reward, next_state):
        if self.cnt_experience % self.update_experience_interval == 0:
            state = state[-self.observation_size:]
            next_state = next_state[-self.observation_size:]
            self.experiences.append((state, action, reward, next_state))
            if len(self.experiences) > self.max_experience_size:
                self.experiences.popleft()
        self.cnt_experience += 1
            
    def update_model(self):
        if self.cnt_step % self.update_model_interval == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        self.cnt_step += 1
        
    def learn(self):
        if len(self.experiences) < self.batch_size:
            return
        
        indices = torch.randint(len(self.experiences), (self.batch_size,))
        samples = [self.experiences[i] for i in indices]
        
        states              = torch.empty((len(samples), self.observation_size)).to(self.device)
        actions             = torch.empty((len(samples), self.strategy_count),dtype=torch.int64).to(self.device)
        rewards             = torch.empty((len(samples), )).to(self.device)
        next_states         = torch.empty((len(samples), self.observation_size)).to(self.device)
        next_states_mask    = torch.empty((len(samples), )).to(self.device)
        for i, (state, action, reward, next_state) in enumerate(samples):
            states[i]       =  torch.as_tensor(state,dtype=torch.float32)
            actions[i]      =  one_hot(torch.as_tensor(action[0],dtype=torch.int64),num_classes = 2)
            rewards[i]      =  torch.as_tensor(float(reward),dtype=torch.float32)
            if next_state is not None:
                    next_states[i] = torch.as_tensor(next_state,dtype=torch.float32)
                    next_states_mask[i] = 1
            else:
                    next_states[i] = 0
                    next_states_mask[i] = 0
            
        q_eval = self.eval_model(states).gather(1, actions)
        q_next = self.target_model(next_states).detach()
        q_eval = torch.sum(q_eval, 1)
        q_target = rewards + self.reward_decay_ratio * (torch.max(q_next, 1)[0] * next_states_mask)
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_model()
        
    def save_model(self, path):
        torch.save(self.target_model.to('cpu').state_dict(), path)
        
    def load_model(self, path):
        self.target_model.load_state_dict(torch.load(path))
        self.eval_model.load_state_dict(self.target_model.to(self.device).state_dict())
        