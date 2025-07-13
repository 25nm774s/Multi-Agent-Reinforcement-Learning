import random
import torch
from collections import deque
import numpy as np
from typing import Deque, Tuple, List, Any, Optional

class ReplayBuffer:
    """
    経験を保存し、学習のためにバッチとして提供するリプレイバッファクラス.
    DQNでの利用を想定しており、経験はPyTorchテンソルに変換されて提供される.
    Prioritized Experience Replay (PER) をサポート.
    """
    def __init__(self, learning_mode: str, buffer_size: int, batch_size: int, device: torch.device, alpha: float = 0.6):
        """
        ReplayBuffer クラスのコンストラクタ.

        Args:
            learning_mode (str): 学習モードを指定する文字列 ('DQN_else'などを想定). 現在の設計ではテンソルを返す動作に影響しないが、元の構造に合わせて保持.
            buffer_size (int): リプレイバッファの最大容量.
            batch_size (int): get_batchでサンプリングするバッチサイズ.
            device (torch.device): データを配置するデバイス (CPUまたはGPU).
            alpha (float): PERの優先度サンプリングのパラメータ (0で一様サンプリング, 1で完全に優先度ベース).
        """
        self.learning_mode: str = learning_mode
        # buffer は経験を格納 (global_state, action, reward, next_global_state, done)
        self.buffer: Deque[Tuple[Any, int, float, Any, bool]] = deque(maxlen=buffer_size)
        # priorities は各経験に対応する優先度を格納
        self.priorities: Deque[float] = deque(maxlen=buffer_size)

        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self.alpha: float = alpha # PER alpha parameter
        # self.beta: float = beta # PER beta parameter (handled in sampling method)

        # 経験が初めて追加される際の初期優先度
        self._max_priority = 1.0 # 最初は最大TD誤差が不明なため、高い値を設定


    def add(self, global_state: Any, action: int, reward: float, next_global_state: Any, done: bool) -> None:
        """
        Add a single experience tuple to the replay buffer with a maximum priority.

        Args:
            global_state (Any): 環境の現在の全体状態 (タプル形式を想定).
            action (int): エージェントが取った行動.
            reward (float): 行動によって得られた報酬.
            next_global_state (Any): 環境の次の全体状態 (タプル形式を想定).
            done (bool): エピソードが完了したかどうかのフラグ.
        """
        data: Tuple[Any, int, float, Any, bool] = (global_state, action, reward, next_global_state, done)
        self.buffer.append(data)
        # 新しい経験には最大の優先度を割り当て、少なくとも一度はサンプリングされるようにする
        self.priorities.append(self._max_priority)


    def __len__(self) -> int:
        """
        Return the current size of the replay buffer.
        """
        return len(self.buffer)

    def sample(self, beta: float) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]]:
        """
        Sample a batch of experiences using prioritized sampling and convert them to PyTorch tensors.
        Also returns sampling probabilities and importance sampling weights.

        Args:
            beta (float): PERの重要度サンプリングのパラメータ (0で重みなし, 1で完全補正).

        Returns:
            A tuple containing tensors for global_states, actions, rewards, next_global_states,
            done flags, importance sampling weights, and a list of sampled indices.
            Returns None if buffer size is less than batch size.
        """
        buffer_len = len(self.buffer)
        if buffer_len < self.batch_size:
            return None

        # 優先度をべき乗して、サンプリング確率を計算
        # priorities_np = np.array(self.priorities)
        # adjusted_priorities = priorities_np**self.alpha
        # sampling_probs = adjusted_priorities / adjusted_priorities.sum()

        # SumTreeやSegment Treeを使用すると効率的だが、dequeの場合は単純なリスト/Numpy配列で計算
        priorities_list = list(self.priorities)
        # 優先度を小さすぎる値でクリップして、ゼロ優先度を避ける
        min_priority = 1e-6
        adjusted_priorities = np.array([max(p, min_priority) for p in priorities_list])**self.alpha
        sum_priorities = adjusted_priorities.sum()
        sampling_probs = adjusted_priorities / sum_priorities


        # 優先度に基づいてバッチサイズ分のインデックスをサンプリング
        # replace=False で非復元抽出
        try:
            sampled_indices: List[int] = random.choices(range(buffer_len), weights=sampling_probs, k=self.batch_size)
        except ValueError as e:
             print(f"Error during random.choices sampling: {e}")
             print(f"Buffer length: {buffer_len}, Batch size: {self.batch_size}")
             print(f"Sampling probabilities sum: {sampling_probs.sum()}")
             # デバッグ用に確率分布や優先度を表示しても良い
             # print("Priorities:", priorities_list)
             # print("Sampling Probs:", sampling_probs)
             return None


        # サンプリングされたインデックスに対応する経験データを抽出
        sampled_experiences = [self.buffer[i] for i in sampled_indices]

        # 重要度サンプリング (IS) 重みの計算
        # Wi = (1/N * 1/P(i))**beta / max(Wi)
        # P(i) は経験 i がサンプリングされる確率 (sampling_probs[i])
        # N はバッファサイズ (buffer_len)
        # is_weights = ((1. / buffer_len) * (1. / sampling_probs[sampled_indices]))**beta
        # 最大重みで正規化 (安定化のため)
        # is_weights /= is_weights.max()

        # 重要度サンプリング (IS) 重みの計算
        # IS_weights = (N * P(i)) ^ -beta
        # P(i) は経験 i がサンプリングされる確率 (sampling_probs[sampled_indices])
        # N はバッファサイズ (buffer_len)
        # is_weights = (buffer_len * sampling_probs[sampled_indices]) ** -beta
        # # 最大重みで正規化 (安定化のため)
        # max_is_weight = (buffer_len * np.min(sampling_probs))**(-beta) # 最小確率の経験が最大重みを持つ
        # is_weights /= max_is_weight

        # 重要度サンプリング (IS) 重みの計算 (より一般的な形式)
        # Wi = (1/N * 1/P(i))**beta
        # P(i) = priorities[i]**alpha / sum(priorities**alpha)
        # Wi = (1/N * sum(priorities**alpha) / priorities[i]**alpha)**beta
        # より簡単な計算: P(i) は sampling_probs から得られる
        # Wi = (1.0 / (buffer_len * sampling_probs[sampled_indices])) ** beta
        # 最大重みで正規化 (安定化のため)
        # max_prob = np.max(sampling_probs[sampled_indices]) if np.max(sampling_probs[sampled_indices]) > 0 else 1e-6 # ゼロ除算対策
        # max_is_weight_unnormalized = (1.0 / (buffer_len * np.min(sampling_probs[sampled_indices]))) ** beta if np.min(sampling_probs[sampled_indices]) > 0 else 1.0 # ゼロ除算対策
        # is_weights /= max_is_weight_unnormalized


        # 重要度サンプリング (IS) 重みの計算 (論文[1]に基づく形式)
        # Wi = (buffer_len * P(i)) ^ -beta
        # P(i) = sampling_probs[sampled_indices]
        is_weights = (buffer_len * sampling_probs[sampled_indices]) ** -beta
        # 正規化: max(Wi) で割る
        # 論文では max(Wi) は経験バッファ全体での最大重みを使用することが多いが、
        # サンプリングされたバッチ内での最大値で正規化することも一般的。
        # ここではバッチ内の最大値で正規化する。
        max_weight = np.max(is_weights) if np.max(is_weights) > 0 else 1.0 # ゼロ除算対策
        is_weights /= max_weight


        # 各要素を抽出 and convert to NumPy arrays with specific dtypes
        try:
            global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[0]]).astype(np.float32) for x in sampled_experiences])
            actions_np: np.ndarray = np.array([x[1] for x in sampled_experiences], dtype=np.int64)
            reward_np: np.ndarray = np.array([x[2] for x in sampled_experiences], dtype=np.float32)
            next_global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[3]]).astype(np.float32) for x in sampled_experiences])
            done_np: np.ndarray = np.array([x[4] for x in sampled_experiences], dtype=np.float32)
            is_weights_np: np.ndarray = np.array(is_weights, dtype=np.float32)

        except Exception as e:
            print(f"Error converting sampled data to numpy arrays: {e}")
            # デバッグのために元のデータを表示しても良い
            # print("Sampled data causing error:", sampled_experiences[0])
            return None


        # NumPy配列をテンソルに変換し, self.device に移動
        global_states_tensor: torch.Tensor = torch.tensor(global_states_np, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.tensor(actions_np, dtype=torch.int64, device=self.device)
        reward_tensor: torch.Tensor = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
        next_global_states_tensor: torch.Tensor = torch.tensor(next_global_states_np, dtype=torch.float32, device=self.device)
        done_tensor: torch.Tensor = torch.tensor(done_np, dtype=torch.float32, device=self.device)
        is_weights_tensor: torch.Tensor = torch.tensor(is_weights_np, dtype=torch.float32, device=self.device)

        # サンプリングされたインデックスも返す (優先度更新のため)
        return global_states_tensor, actions_tensor, reward_tensor, next_global_states_tensor, done_tensor, is_weights_tensor, sampled_indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update the priorities of the experiences at the given indices.

        Args:
            indices (List[int]): The indices of the experiences to update.
            td_errors (np.ndarray): The new TD errors for the experiences (absolute values).
        """
        # 小さすぎるTD誤差で優先度がゼロにならないようにクリップ
        min_error = 1e-6
        # 優先度は |TD error| に比例
        new_priorities = np.maximum(np.abs(td_errors), min_error)

        for idx, priority in zip(indices, new_priorities):
            # deque の要素を直接更新することはできないため、新しい優先度を持つ要素で置き換えるか、
            # 内部的にリストや別のデータ構造で優先度を管理する必要がある。
            # 簡単のため、ここではリストに変換して更新し、再度dequeに戻す (効率は悪いが動作確認用)
            # より効率的な実装には SumTree などが必要
            # TODO: deque の代わりに SumTree または Segment Tree を使用して効率化
            # 現在の実装では、大きなバッファサイズの場合に優先度更新が遅くなる可能性がある

            # Workaround for deque: convert to list, update, convert back
            priorities_list = list(self.priorities)
            priorities_list[idx] = priority
            self.priorities = deque(priorities_list, maxlen=self.buffer_size)

            # 最大優先度を更新
            self._max_priority = max(self._max_priority, priority)

    # Note: Removed the old get_batch method with learning_mode branching
    # If 'V' or 'Q' learning modes are needed later, dedicated methods should be added.
    # Renamed get_dqn_batch to sample for clarity with PER