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
    def __init__(self, args):
        self.agents_num = args.agents_number
        self.batch_size = args.batch_size
        self.decay_epsilon_step = args.decay_epsilon
        self.action_size = 5
        self.epsilon = None
        #self.learning_mode = args.learning_mode
        self.load_model = args.load_model
        self.goals_num = args.goals_number
        self.mask = args.mask
        #self.model_path = agentID
        self.device = torch.device(args.device)
        self.replay_buffer = ReplayBuffer("Q",args.buffer_size,self.batch_size,self.device)
        self.linear = Linear(args,5)

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
    # 非推奨関数
    def update_brain(self, i, states, action, reward, next_state, done, episode_num, step):
        self.replay_buffer.add(states, action, reward, next_state, done)

        if self.load_model == 1:
            scalar_loss = None # 学習済みモデル使用時, 更新しない

        if len(self.replay_buffer) < self.batch_size:
            return None
        
        states, action, reward, next_state, done = self.replay_buffer.get_batch()

        # 線形関数近似器
        scalar_loss = []
        # 修正：ループ範囲を実際のバッチサイズ len(states) に変更
        for j in range(len(states)): # ここではバッチサイズ分のlossを平均
            scalar_loss.append(self.linear.update(i, states[j], action[j], reward[j], next_state[j], done[j], step))
        scalar_loss = np.mean(scalar_loss)

        return scalar_loss
    
    def observe_and_store_experience(self, state, action, reward, next_state, done):
        """
        環境からの単一ステップの経験をリプレイバッファに追加する。
        """
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn_from_experience(self, i, episode_num):
        """
        リプレイバッファからバッチを取得し、モデルを学習させる。
        """
        if self.load_model == 1:
            return None # 学習済みモデル使用時は更新なし

        #if len(self.replay_buffer) < self.model.batch_size: # batch_sizeはmodelにあると仮定
        if len(self.replay_buffer) < self.batch_size: # batch_sizeはselfにある
            return None # バッチサイズに満たない場合は学習しない

        # 1. バッチデータの取得
        states, action, reward, next_state, done = self.replay_buffer.get_batch()
        
        # 2. モデルの更新 episode_numはターゲットネットワーク更新タイミングのため必要
        # 線形関数近似器
        scalar_loss = []
        # 修正：ループ範囲を実際のバッチサイズ len(states) に変更
        for j in range(len(states)): # ここではバッチサイズ分のlossを平均
            scalar_loss.append(self.linear.update(i, states[j], action[j], reward[j], next_state[j], done[j], step=episode_num))
        scalar_loss = np.mean(scalar_loss)

        return scalar_loss