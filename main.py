import torch
from itertools import product


from utils.runners import NPlayerEnv
from players.ml_intelligence import DQNPlayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init():
    print('请输入博弈参与者，使用英文逗号(,)分隔 : ', end='\x1b[35m')                                                                                 
    player_names = list(str(input()).split(','))
    print('\x1b[0m',end='')
    players = []
    strategies_list = []
    strategies_cnt = []
    strategies_total_prod = 1
    
    for player in player_names:
        print('请输入博弈参与者 \x1b[32m{}\x1b[0m 的策略集，使用英文逗号(,)分隔 : '.format(player), end='\x1b[35m')
        strategies = list(str(input()).split(','))
        print('\x1b[0m',end='')
        strategies_list.append(strategies)
        strategies_cnt.append(len(strategies))
        strategies_total_prod *= len(strategies)
        players.append(DQNPlayer(strategies,len(player_names)-1, device))
    cost_matrix_str = '(' + '0.,'*len(player_names) + ')'
    for item in strategies_cnt[::-1]:
        tmp_str = cost_matrix_str + ','
        tmp_str = tmp_str * item
        cost_matrix_str = '[' + tmp_str + ']'
    cost_matrix = torch.tensor(eval(cost_matrix_str))
    for i in range(strategies_total_prod):
        strategy_permutation = ['0'] * len(player_names)
        curr_indice = 0
        tt_i = i
        while tt_i > 0:
            strategy_permutation[curr_indice] = str(tt_i%strategies_cnt[curr_indice])
            tt_i //= strategies_cnt[curr_indice]
            curr_indice += 1
        valid_flag = False
        cost_vec = torch.zeros(len(strategy_permutation), dtype=torch.float32)
        while not valid_flag:
            print('请输入 ',end='')
            for i,item in enumerate(strategy_permutation):
                item = int(item)
                print('{}\x1b[32m{}\x1b[0m 选择 \x1b[33m{}\x1b[0m '.format('' if i == 0 else ',', 
                                                player_names[i], 
                                                strategies_list[i][item]
                                            ),end='')

            print('情况下的价值向量，使用英文逗号(,)分隔 : ',end='\x1b[35m')
            vec_list = list(map(float, str(input()).split(',')))
            print('\x1b[0m',end='')
            if len(vec_list) == len(strategy_permutation):
                cost_vec = torch.tensor(vec_list)
                valid_flag = True
            else:
                print('价值向量输入无效，请重新输入!') 
        cost_matrix[eval(','.join(strategy_permutation))] = cost_vec   
    return player_names, players,strategies_list, cost_matrix  
            
     
def decide(player_names, players, strategies_list):
    
    player_id = 0
    flag = False
    while not flag:
    
        print('请输入您要测试的一方(请在\x1b[32m{}\x1b[0m中选择) : '.format(player_names),end='\x1b[35m')
        player_str = input() 
        print('\x1b[0m',end='')
        if player_names.count(player_str):
            flag=True
            player_id = player_names.index(player_str)
        else:
            print('策略参与者输入无效，请重新输入!') 
        
        
    player_opp = torch.zeros(len(player_names) - 1)
    curr_indice = 0
    for i, item in enumerate(player_names):
        if i == player_id:
            continue
        flag = False
        while not flag:
            print('请输入参与者 \x1b[32m{}\x1b[0m 的选择(请在\x1b[33m{}\x1b[0m中选择) : '.format(item, strategies_list[i]),end='\x1b[35m')
            choice = input()
            print('\x1b[0m',end='')
            if strategies_list[i].count(choice):
                flag = True
                player_opp[curr_indice] = strategies_list[i].index(choice)
                curr_indice += 1
            else:
                print('策略选择输入无效，请重新输入!') 
                
    print('参与者 \x1b[32m{}\x1b[0m 的选择是 : \x1b[34m{}\x1b[0m'.format(player_names[player_id], players[player_id].decide(player_opp)[1][0]))    
    
player_names, players,strategies_list, cost_matrix = init()
env = NPlayerEnv(players,player_names,cost_matrix)
env.activate(load=False)

print('请输入要迭代的次数 : ',end='\x1b[35m')
steps = int(input())
print('\x1b[0m',end='')
env.play(steps)

print('训练完成，是否保存[yes, no] :',end='\x1b[35m')
save = input()
print('\x1b[0m',end='')
env.freeze(True if save == 'yes' else False)

decide(player_names, players,strategies_list)
