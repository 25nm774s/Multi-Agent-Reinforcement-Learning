"""
リプレイバッファ.
経験を溜め, 出力時に各種変数をテンソルに変換.
"""

import random
import torch
from collections import deque
import numpy as np
from typing import Deque, Tuple, List, Any, Optional

# NNモデル使用時
class ReplayBuffer:
    """
    経験を保存し、学習のためにバッチとして提供するリプレイバッファクラス.
    DQNでの利用を想定しており、経験はPyTorchテンソルに変換されて提供される.
    """
    def __init__(self, learning_mode: str, buffer_size: int, batch_size: int, device: torch.device):
        """
        ReplayBuffer クラスのコンストラクタ.

        Args:
            learning_mode (str): 学習モードを指定する文字列 ('DQN_else'などを想定). 現在の設計ではテンソルを返す動作に影響しないが、元の構造に合わせて保持.
            buffer_size (int): リプレイバッファの最大容量.
            batch_size (int): get_batchでサンプリングするバッチサイズ.
            device (torch.device): データを配置するデバイス (CPUまたはGPU).
        """
        self.learning_mode: str = learning_mode # get_batch の分岐に使用される想定 (今回は主にget_dqn_batchを使用)
        # Define the type of elements stored in the deque
        # Each element is a tuple: (global_state, action, reward, next_global_state, done)
        # global_state and next_global_state are expected to be tuples of positions (e.g., tuples of tuples)
        # action is an int
        # reward is a float
        # done is a bool
        # Using Tuple[Any, int, float, Any, bool] for flexibility, assuming global_state/next_global_state are complex structures like tuples of tuples.
        # If states are always tuples of tuples of ints (e.g., ((gx1, gy1), ...), ((ax1, ay1), ...)), a more precise hint could be used.
        # For now, use Any for global_state/next_global_state components within the tuple.
        self.buffer: Deque[Tuple[Any, int, float, Any, bool]] = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.device: torch.device = device


    def add(self, global_state: Any, action: int, reward: float, next_global_state: Any, done: bool) -> None:
        """
        Add a single experience tuple to the replay buffer.

        Args:
            global_state (Any): 環境の現在の全体状態 (タプル形式を想定).
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_global_state (Any): 環境の次の全体状態 (タプル形式を想定).
            done (bool): エピソードが完了したかどうかのフラグ.
        """
        data: Tuple[Any, int, float, Any, bool] = (global_state, action, reward, next_global_state, done)
        self.buffer.append(data)

    def __len__(self) -> int:
        """
        Return the current size of the replay buffer.
        """
        return len(self.buffer)

    def get_dqn_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Sample a batch of experiences specifically for DQN and convert them to PyTorch tensors.

        Returns:
            A tuple containing tensors for global_states, actions, rewards, next_global_states, and done flags.
            Returns None if buffer size is less than batch size.
        """
        if len(self.buffer) < self.batch_size:
            #raise
            return None # バッチサイズに満たない場合は None を返す

        data: List[Tuple[Any, int, float, Any, bool]] = random.sample(self.buffer, self.batch_size)

        # 各要素を抽出 and convert to NumPy arrays with specific dtypes
        # x is a tuple: (global_state, action, reward, next_global_state, done)
        # x[0] is global_state (e.g., tuple of tuples)
        # np.concatenate(x[0]) attempts to flatten the global_state structure
        # Assuming global_state structure is flattenable into a 1D array of floats.
        # 例: global_state = ((gx1, gy1), ..., (ax1, ay1), ...), np.concatenate(global_state) -> [gx1, gy1, ..., ax1, ay1, ...]
        # Note: This assumes global_state is a tuple/list of tuples/lists of numbers.
        # If the structure is different, this conversion might need adjustment.
        try:
            global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[0]]).astype(np.float32) for x in data])
            actions_np: np.ndarray = np.array([x[1] for x in data], dtype=np.int64) # 行動はエージェントiの行動を想定 (Agent_DQN.observe_and_store_experience で単一エージェントの行動がaddされている場合)
            reward_np: np.ndarray = np.array([x[2] for x in data], dtype=np.float32) # 報酬は全体報酬または個別報酬を想定
            next_global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[3]]).astype(np.float32) for x in data])
            done_np: np.ndarray = np.array([x[4] for x in data], dtype=np.float32) # Doneフラグはfloatで扱うことが多い (1.0 or 0.0)
        except Exception as e:
            print(f"Error converting batch data to numpy arrays: {e}")
            # データ構造が期待と異なる場合にエラーになる可能性
            # デバッグのために元のデータを表示しても良い
            # print("Sample data causing error:", data[0])
            #raise
            return None


        # NumPy配列をテンソルに変換し, self.device に移動
        global_states_tensor: torch.Tensor = torch.tensor(global_states_np, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.tensor(actions_np, dtype=torch.int64, device=self.device)
        reward_tensor: torch.Tensor = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
        next_global_states_tensor: torch.Tensor = torch.tensor(next_global_states_np, dtype=torch.float32, device=self.device)
        done_tensor: torch.Tensor = torch.tensor(done_np, dtype=torch.float32, device=self.device) # Doneフラグをfloat32に変換

        return global_states_tensor, actions_tensor, reward_tensor, next_global_states_tensor, done_tensor

    # Note: Removed the old get_batch method with learning_mode branching
    # If 'V' or 'Q' learning modes are needed later, dedicated methods should be added.