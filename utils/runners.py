import os
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from utils.base import EnvBase


class NPlayerEnv(EnvBase):
    def __init__(self, players : list, player_names : list, cost_matrix : list):
        super(NPlayerEnv, self).__init__()
        if not isinstance(players, list) or len(players) < 2:
            raise Exception('博弈参与者人数至少为2')
        if len(players) != len(player_names):
            raise Exception('博弈参与实例个数与博弈参与者名称列表长度不一致')
        
        self.player_names = player_names
        self.players = {}
        for i, item in enumerate(self.player_names):
            self.players[item] = players[i]
        self.cost_matrix = cost_matrix
        self.model_paths = []
        self.actions = {}
        self.rewards = {}
        self.data = {}
        
        self.activate_status = False
        
    def activate(self, model_paths = None, load = True):
        self.model_paths = []
        self.actions = {}
        self.rewards = {}
        self.data = {}
        for player in self.player_names:
            self.data[player] = []
            
        if model_paths is None:
            for player in self.player_names:
                self.model_paths.append('./saved_models/{}.pt'.format(player))
                
        else:
            curr_len = len(model_paths)
            full_len = len(self.player_names)
            if curr_len != full_len:
                raise Exception('路径和博弈参与者列表长度不一致')
            for model_path in model_paths:
                self.model_paths.append(model_path)
                
            for i, item in enumerate(self.model_paths):
                if item is None:
                    self.model_paths[i] = './saved_models/{}.pt'.format(player)
                else:
                    if load == True:
                        self.players[self.player_names[i]].load_model(item)
                                      
        self.activate_status = True
        
    def save_checkpoint(self, *args, **kwargs):
        if  self.activate_status == False:
            return
        for i, item in enumerate(self.model_paths):
            self.players[self.player_names[i]].save_model(item)
    
    def freeze(self, save = True, *args, **kwargs):
        self.activate_status = False
        if save == True:
            for i, item in enumerate(self.model_paths):
                self.players[self.player_names[i]].save_model(item)
    
    def play(self, steps=1000, *args, **kwargs):
        if self.activate_status == False:
            return
        
        with Progress(TextColumn("[progress.description]{task.description}"),
              BarColumn(bar_width=50),
              TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
              TimeRemainingColumn(),
              TimeElapsedColumn()) as progress:
            game_process = progress.add_task(description="博弈迭代过程", total=steps)
            player_process = progress.add_task(description="博弈参与过程", total=len(self.players))
            for _ in range(steps):
                progress.advance(game_process, 1)
                
                for player in self.player_names:
                    progress.advance(player_process, 1) 
                    self.actions[player] = self.players[player].decide(state=self.data[player])
                progress.reset(player_process)
                    
                self._judge()
                
                for player in self.player_names:
                    progress.advance(player_process, 1) 
                    self.players[player].learn(self.data[player][:-1],
                                               self.actions[player][0],
                                               self.rewards[player],
                                               self.data[player]
                                              )
                progress.reset(player_process)
                        
    def _judge(self, *args, **kwargs):
        permutation_str = 'self.cost_matrix' 
        for dvalue in self.actions.values():
            permutation_str += '{}'.format(dvalue[0])
        cost_values = eval(permutation_str)
        
        for i, reward in enumerate(cost_values):
            self.rewards[self.player_names[i]] = reward
            self.data[self.player_names[i]].append(reward)
            self.players[self.player_names[i]].settle(reward)
        