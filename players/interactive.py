from players.base import PlayerBase


class Interactive(PlayerBase):
    """可交互的人类博弈参与者
    
    """
    def __init__(self, strageties : list, initial_score=1., mul_ratio=1.):
        super(Interactive, self).__init__(strageties, initial_score, mul_ratio)
        
    def decide(self):
        strategie_select = input()
        try:
            strategie_index = int(strategie_select)
            return (strategie_index, self.strategies[strategie_index])
        except:
            if self.strategies.count(strategie_select):
                strategie_index = self.strategies.index(strategie_select)
                return (strategie_index, self.strategies[strategie_index])
            else:
                return (0, self.strategies[0])
        
    def settle(self, val):
        return self._reward(val) if val > 0 else self._punish(val)
    