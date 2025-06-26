"""
エージェントの行動を出力.
学習器の更新など.
"""

import torch
import numpy as np
import os
import sys

#from core.linear import Linear
#from core.replay_buffer import ReplayBuffer
#from core.dqn import DQN
from Q_learn.linear import Linear
from utils.replay_buffer import ReplayBuffer

np.random.seed(0)
torch.manual_seed(0)

# ターミナルの表示関連
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

MAX_EPSILON = 1
MIN_EPSILON = 0.01

class Agent_Q:
    def __init__(self, args, model_path):
        self.agents_num = args.agents_number
        self.batch_size = args.batch_size
        self.decay_epsilon_step = args.decay_epsilon
        self.action_size = 5
        self.epsilon = None
        #self.learning_mode = args.learning_mode
        self.load_model = args.load_model
        self.goals_num = args.goals_number
        self.mask = args.mask
        self.model_path = model_path
        self.device = torch.device(args.device)
        self.replay_buffer = ReplayBuffer("Q",args.buffer_size,self.batch_size,self.device)

        # モデルのロード
        if self.load_model == 1 or self.load_model == 2:
            self.model_path = model_path
            self.loading_model(self.model_path)

        self.linear = Linear(args, self.action_size, self.model_path, "Q")

    # 学習済みモデルの存在の確認
    def loading_model(self, model_path):
        if os.path.exists(model_path):
            print('モデルを読み込みました.')
            print(f"from {GREEN}{model_path}{RESET}\n")
        else:
            print(f"学習済みモデル {RED}{model_path}{RESET} が見つかりません.")
            print('学習する場合, load_model=0 に変更してください.\n')
            sys.exit()

    def get_action(self, i, states):    
        return self.linear_greedy_actor(i, states)

    # 線形関数近似のε-greedy
    def linear_greedy_actor(self, i, states):
        goals_pos = [list(pos) for pos in states[:self.goals_num]]#なぜか不使用変数
        agents_pos = [list(pos) for pos in states[self.goals_num:]]
        agent_pos = agents_pos[i]

        agents_pos.pop(i)

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax([self.linear.getQ(agents_pos, agent_pos, action) for action in range(self.action_size)])

    # Q系列のNNモデル使用時のε-greedy
    def nn_greedy_actor(self, i, states):
        if self.mask:
            states = states[self.goals_num + i]

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            # stateがタプルの場合の整形
            if isinstance(states, tuple):
                flat_state = np.array(states).flatten()
                states = torch.tensor(flat_state, dtype=torch.float32) # 1次元のテンソルに変換
            elif isinstance(states, torch.Tensor):
                pass

            states = states.to(self.device)
            qs = self.model.qnet(states)
            return qs.argmax().item()

    # epsilonの線形アニーリング
    def decay_epsilon(self, step):
        if step < self.decay_epsilon_step:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.decay_epsilon_step - step) / self.decay_epsilon_step

    # 価値更新
    def update_brain(self, i, states, action, reward, next_state, done, episode_num, step):
        self.replay_buffer.add(states, action, reward, next_state, done)

        if self.load_model == 1:
            scalar_loss = None # 学習済みモデル使用時, 更新しない

        if len(self.replay_buffer) < self.batch_size:
            return None
            #return 
        
        states, action, reward, next_state, done = self.replay_buffer.get_batch()

        # 線形関数近似器
        scalar_loss = []
        # 修正：ループ範囲を実際のバッチサイズ len(states) に変更
        for j in range(len(states)): # ここではバッチサイズ分のlossを平均
            scalar_loss.append(self.linear.update(i, states[j], action[j], reward[j], next_state[j], done[j], step))
        scalar_loss = np.mean(scalar_loss)

        return scalar_loss