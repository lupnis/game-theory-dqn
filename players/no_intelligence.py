import random

from players.base import PlayerBase


class Chaos(PlayerBase):
    """随机做决策的决策者

    """
    def __init__(self, strageties : list, initial_score=1., mul_ratio=1., *args, **kwargs):
        super(Chaos, self).__init__(strageties, initial_score, mul_ratio)
        
    def decide(self, *args, **kwargs):
        strategie_index = int(random.uniform(0,len(self.strategies) - 1))
        return (strategie_index, self.strategies[strategie_index])
    
    def settle(self, val, *args, **kwargs):
        return self._reward(val) if val > 0 else self._punish(val)
    
    
class Stubborn(PlayerBase):
    """只做一种决策的决策者

    """
    def __init__(self, strageties : list, fixed_index = 0, initial_score=1., mul_ratio=1., *args, **kwargs):
        super(Stubborn, self).__init__(strageties, initial_score, mul_ratio)
        self.fixed_index = fixed_index
    
    def decide(self, *args, **kwargs):
        return (int(self.fixed_index), int(self.strategies[self.fixed_index]))
    
    def settle(self, val, *args, **kwargs):
        return self._reward(val) if val > 0 else self._punish(val)
    
    
class Preferred(PlayerBase):
    """偏向某种决策的决策者
    
    """
    def __init__(self, strageties : list, preferred_index = 0, preferred_prob = 0.5, initial_score=1., mul_ratio=1., *args, **kwargs):
        super(Preferred, self).__init__(strageties, initial_score, mul_ratio)
        self.preferred_index = preferred_index
        self.preferred_prob = preferred_prob
    
    def decide(self, *args, **kwargs):
        pre_prob = random.uniform(0, 1)
        if pre_prob > self.preferred_prob:
            return (int(self.preferred_index), self.strategies[int(self.preferred_index)])
        else:
            strategie_index = int(random.uniform(1,len(self.strategies) - 1))
            return (strategie_index, self.strategies[strategie_index])
            
    def settle(self, val, *args, **kwargs):
        return self._reward(val) if val > 0 else self._punish(val)
