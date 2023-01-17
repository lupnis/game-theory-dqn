from torch import as_tensor


class PlayerBase(object):
    """博弈参与者基类
    
       本基类仅定义博弈参与者所需要的各种参数
       
    """
    def __init__(self, strageties : list, initial_score, mul_ratio, *args, **kwargs):
        """博弈参与者基类初始化函数

        Args:
            strategies (list): 博弈者策略集
            initial_score (torch.Tensor): 博弈参与者的初始分数
            mul_ratio (torch.Tensor): 博弈者得分/罚分接受比率系数
            
        """
        self.strategies     = strageties
        self.initial_score  = as_tensor(initial_score)
        self.mul_ratio      = as_tensor(mul_ratio)
        
    def __call__(self, *args, **kwargs):
        """博弈参与者输出选择策略函数
        
        Returns:
            tuple(int, str): 返回`(策略编号, 策略名称)`元组
            
        """
        return self.decide(*args, **kwargs)
        
    def decide(self, *args, **kwargs) -> tuple:
        """博弈参与者输出选择策略函数
        
        Returns:
            tuple(int, str): 返回`(策略编号, 策略名称)`元组
            
        """
        ...
    
    def settle(self, val, *args, **kwargs):
        """博弈参与者选择策略后结算并选用奖励/惩罚函数
        Args:
            val (torch.float32): 得分值，用于进行奖励或惩罚决定
            
        """
        ...
    
    def _reward(self, val, *args, **kwargs) -> tuple:
        """增加奖励分函数

        Args:
            val (torch.Tensor): 奖励初始值
            
        Returns:
            tuple(Tensor, Tensor):返回`(分数, 总计)`元组
            
        """
        self.initial_score += self.mul_ratio * val
        return val, self.initial_score
    
    def _punish(self, val, *args, **kwargs) -> tuple:
        """扣除惩罚分函数

        Args:
            val (torch.Tensor): 惩罚初始值
        
        Returns:
            tuple(Tensor, Tensor):返回`(分数, 总计)`元组
            
        """
        self.initial_score -= self.mul_ratio * val
        return val, self.initial_score
    
    def learn(self, *args, **kwargs):
        """博弈参与者学习函数
            
        """
        ...
        
    def save_model(self, path, *args, **kwargs):
        """博弈参与者模型保存
        
        Args:
            path (str): 模型保存路径
            
        """
        ...
        
    def load_model(self, path, *args, **kwargs):
        """博弈参与者模型加载
        
        Args:
            path (str): 模型加载路径
        
        """
        ...
