"""
リプレイバッファ.
経験を溜め, 出力時に各種変数をテンソルに変換.
"""

import random
import torch
from collections import deque
import numpy as np


# NNモデル使用時
class ReplayBuffer:
    def __init__(self, args):
        self.learning_mode = args.learning_mode
        self.buffer = deque(maxlen=args.buffer_size)
        self.batch_size = args.batch_size
        self.device = args.device


    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    # トレーニング用のデータをリプレイバッファから取り出し, tensor に変換
    def get_batch(self):
        # data = [(((ゴール座標), (エージェント座標)), 行動, 報酬, ((ゴール座標'), (エージェント座標')), done),...

        if self.learning_mode == 'V':
            data = random.sample(self.buffer, self.batch_size)
            states, actions, reward, next_state, done = zip(*data)
            return states, actions, reward, next_state, done
        
        elif self.learning_mode == 'Q':
            states, actions, reward, next_state, done = self.buffer[-1]
            return [states], [actions], [reward], [next_state], [done]
        
        else:
            data = random.sample(self.buffer, self.batch_size)
            # 各要素を抽出
            states_np = np.array([np.concatenate(x[0]).astype(np.float32) for x in data])
            actions_np = np.array([x[1] for x in data], dtype=np.int64)
            reward_np = np.array([x[2] for x in data], dtype=np.float32)
            next_state_np = np.array([np.concatenate(x[3]).astype(np.float32) for x in data])
            done_np = np.array([x[4] for x in data], dtype=np.int32)

            # NumPy配列をテンソルに変換し, self.device に移動
            states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions_np, dtype=torch.int64, device=self.device)
            reward = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=self.device)
            done = torch.tensor(done_np, dtype=torch.int32, device=self.device)

            return states, actions, reward, next_state, done