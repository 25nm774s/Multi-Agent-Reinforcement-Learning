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
    def __init__(self, learning_mode: str, buffer_size: int, batch_size: int, device: torch.device, alpha: float = 0.6, use_per: bool = False):
        """
        ReplayBuffer クラスのコンストラクタ.

        Args:
            learning_mode (str): 学習モードを指定する文字列 ('DQN_else'などを想定). 現在の設計ではテンソルを返す動作に影響しないが、元の構造に合わせて保持.
            buffer_size (int): リプレイバッファの最大容量.
            batch_size (int): get_batchでサンプリングするバッチサイズ.
            device (torch.device): データを配置するデバイス (CPUまたはGPU).
            alpha (float): PERの優先度サンプリングのパラメータ (0で一様サンプリング, 1で完全に優先度ベース).
            use_per (bool, optional): Prioritized Experience Replay を使用するかどうか. Defaults to False. (Step 1)
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
        self.use_per: bool = use_per # Store use_per flag (Step 1)


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
        if self.use_per:
            self.priorities.append(self._max_priority)
        else:
            # If PER is not used, priority doesn't matter for sampling, but the deque needs an element
            # Append 0.0 or any default value
            self.priorities.append(0.0)


    def __len__(self) -> int:
        """
        Return the current size of the replay buffer.
        """
        return len(self.buffer)

    # Modify sample to handle uniform sampling when use_per is False (Step 3)
    def sample(self, beta: float) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[List[int]]]]:
        """
        Sample a batch of experiences using prioritized sampling (if enabled) or uniform sampling.
        Convert them to PyTorch tensors.
        Also returns sampling probabilities and importance sampling weights (for PER).

        Args:
            beta (float): PERの重要度サンプリングのパラメータ (0で重みなし, 1で完全補正).

        Returns:
            A tuple containing tensors for global_states, actions, rewards, next_global_states,
            done flags. If use_per is True, also returns importance sampling weights tensor
            and a list of sampled indices. If use_per is False, these last two are None.
            Returns None if buffer size is less than batch size.
        """
        buffer_len = len(self.buffer)
        if buffer_len < self.batch_size:
            return None

        if self.use_per:
            # PER sampling logic (existing code)
            priorities_list = list(self.priorities)
            min_priority = 1e-6
            adjusted_priorities = np.array([max(p, min_priority) for p in priorities_list])**self.alpha
            sum_priorities = adjusted_priorities.sum()
            sampling_probs = adjusted_priorities / sum_priorities

            try:
                sampled_indices: List[int] = random.choices(range(buffer_len), weights=sampling_probs, k=self.batch_size)
            except ValueError as e:
                print(f"Error during random.choices sampling (PER): {e}")
                print(f"Buffer length: {buffer_len}, Batch size: {self.batch_size}")
                print(f"Sampling probabilities sum: {sampling_probs.sum()}")
                return None

            # Importance Sampling (IS) weights calculation
            is_weights_np = (buffer_len * sampling_probs[sampled_indices]) ** -beta
            max_weight = np.max(is_weights_np) if np.max(is_weights_np) > 0 else 1.0
            is_weights_np /= max_weight

            is_weights_tensor: torch.Tensor = torch.tensor(is_weights_np, dtype=torch.float32, device=self.device)#type:ignore

        else:
            # Uniform sampling when PER is disabled (Step 3)
            try:
                sampled_indices: List[int] = random.sample(range(buffer_len), k=self.batch_size)
            except ValueError as e:
                print(f"Error during random.sample (Uniform): {e}")
                print(f"Buffer length: {buffer_len}, Batch size: {self.batch_size}")
                return None

            is_weights_tensor: Optional[torch.Tensor] = None # No IS weights for uniform sampling

        # Extract sampled experiences
        sampled_experiences = [self.buffer[i] for i in sampled_indices]

        # Convert to NumPy arrays
        try:
            global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[0]]).astype(np.float32) for x in sampled_experiences])
            actions_np: np.ndarray = np.array([x[1] for x in sampled_experiences], dtype=np.int64)
            reward_np: np.ndarray = np.array([x[2] for x in sampled_experiences], dtype=np.float32)
            next_global_states_np: np.ndarray = np.array([np.concatenate([np.asarray(item) for item in x[3]]).astype(np.float32) for x in sampled_experiences])
            done_np: np.ndarray = np.array([x[4] for x in sampled_experiences], dtype=np.float32)

        except Exception as e:
            print(f"Error converting sampled data to numpy arrays: {e}")
            return None

        # Convert to Tensors
        global_states_tensor: torch.Tensor = torch.tensor(global_states_np, dtype=torch.float32, device=self.device)
        actions_tensor: torch.Tensor = torch.tensor(actions_np, dtype=torch.int64, device=self.device)
        reward_tensor: torch.Tensor = torch.tensor(reward_np, dtype=torch.float32, device=self.device)
        next_global_states_tensor: torch.Tensor = torch.tensor(next_global_states_np, dtype=torch.float32, device=self.device)
        done_tensor: torch.Tensor = torch.tensor(done_np, dtype=torch.float32, device=self.device)

        # Return sampled_indices only if PER is used (Step 3)
        return global_states_tensor, actions_tensor, reward_tensor, next_global_states_tensor, done_tensor, is_weights_tensor, sampled_indices if self.use_per else None


    # Modify update_priorities to do nothing if use_per is False (Step 4)
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        Update the priorities of the experiences at the given indices.
        Does nothing if PER is disabled.

        Args:
            indices (List[int]): The indices of the experiences to update.
            td_errors (np.ndarray): The new TD errors for the experiences (absolute values).
        """
        # Do nothing if PER is disabled (Step 4)
        if not self.use_per:
            return

        # PER priority update logic (existing code)
        min_error = 1e-6
        new_priorities = np.maximum(np.abs(td_errors), min_error)

        for idx, priority in zip(indices, new_priorities):
            # Workaround for deque: convert to list, update, convert back
            priorities_list = list(self.priorities)
            priorities_list[idx] = priority
            self.priorities = deque(priorities_list, maxlen=self.buffer_size)

            # 最大優先度を更新
            self._max_priority = max(self._max_priority, priority)

    # Note: Removed the old get_batch method with learning_mode branching
    # If 'V' or 'Q' learning modes are needed later, dedicated methods should be added.
    # Renamed get_dqn_batch to sample for clarity with PER