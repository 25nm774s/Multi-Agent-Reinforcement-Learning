"""
======================================================================
DQNのネットワークアーキテクチャ.
======================================================================
"""

import torch.nn as nn
import torch.nn.functional as F

import torch
import numpy as np
import torch.optim as optim
    
class QNet(nn.Module):
    def __init__(self, mask, agents_num, goals_num, action_size):
        super().__init__()
        
        if mask:
            input_size = 2
        else:
            input_size = agents_num*2 + goals_num*2

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    # 順伝播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN:
    def __init__(self, args, action_size, model_path):
        agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.lr = args.learning_rate
        self.gamma = args.gamma
        self.load_model = args.load_model
        self.device = args.device
        self.batch_size = args.batch_size
        self.optimizer = args.optimizer
        self.mask = args.mask

        self.qnet = QNet(self.mask, agents_num, self.goals_num, action_size).to(self.device)
        self.qnet_target = QNet(self.mask, agents_num, self.goals_num, action_size).to(self.device)

        # モデルの読み込み
        if self.load_model == 1:
            self.qnet.load_state_dict(torch.load(model_path))
            self.qnet_target.load_state_dict(torch.load(model_path))
            self.qnet.eval()
            self.qnet_target.eval()
        elif self.load_model == 2:
            self.qnet_target.load_state_dict(torch.load(model_path))
            self.qnet_target.eval()

        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        elif self.optimizer == 'RMSProp':
            self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=self.lr)

    def huber_loss(self, q, target):
        err = target - q
        abs_err = abs(err)

        # 標準偏差を用いた動的なHUBER_LOSS_DELTAの設定
        if len(err) > 1:
            huber_loss_delta = torch.mean(abs_err).item() + torch.std(err).item() # 各errの絶対値の平均値 + 標準偏差
        else:
            huber_loss_delta = torch.mean(abs_err).item()  # データが1つしかない場合、標準偏差は計算しない
        
        cond = torch.abs(err) < huber_loss_delta
        L2 = 0.5 * torch.square(err)
        L1 = huber_loss_delta * (torch.abs(err) - 0.5 * huber_loss_delta)
        loss = torch.where(cond, L2, L1)

        return torch.mean(loss)

    # ターゲットネットワークの更新
    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    # モデルアップデート
    def update(self, i, states, action, reward, next_state, done, episode_num):

        if self.mask:
            idx = i*2 + self.goals_num*2
            idx_lst = [idx, idx + 1]
            states = states[:, idx_lst]
            next_state = next_state[:, idx_lst]

        if self.load_model == 0 or self.load_model == 1: # 未学習と学習済みモデル使用時

            qs = self.qnet(states) # 各 state におけるQ値
            q = qs[np.arange(self.batch_size), action] # 実際に取った行動

            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(1)[0] # 各 next_state において最も値の高いQ値

            next_q.detach()
            target = reward + (1 - done) * self.gamma * next_q

            loss = self.huber_loss(q, target)

            self.optimizer.zero_grad() # 累積勾配にならないように初期化
            loss.backward() # 勾配導出と逆伝播
            self.optimizer.step() # パラメータ更新

            #scalar_loss = loss.item()
            scalar_loss = (target - q).mean().item()

            if episode_num % 100 == 0:
                self.sync_qnet() # 100 episode 毎にターゲットネットワーク更新
        
        else: # 学習済みモデルを真の価値関数として改めて学習

            qs = self.qnet(states)
            q = qs[np.arange(self.batch_size), action]

            true_qs = self.qnet_target(states)
            true_q = true_qs[np.arange(self.batch_size), action]

            loss = self.huber_loss(q, true_q)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #scalar_loss = loss.item()    
            scalar_loss = (true_q - q).mean().item()


        return scalar_loss