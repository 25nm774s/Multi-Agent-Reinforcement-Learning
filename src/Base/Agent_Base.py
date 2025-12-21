import torch
from typing import Tuple, List, Any
from abc import ABC, abstractmethod

# --- Type Aliases ---
# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)
PositionType = Tuple[int, int]
# QTableType はQTableクラスの実際の戻り値型と一致させる

# --- Global Constants ---
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

class AgentBase(ABC):
    """
    Q学習エージェントとDQNエージェントに共通のインターフェースを定義する抽象基底クラス.
    """
    def __init__(self, agent_id: int, args):
        self.agent_id:int           = agent_id
        self.grid_size:int          = args.grid_size
        self.goals_num:int          = args.goals_number
        self.action_size:int        = 5 # UP, DOWN, LEFT, RIGHT, STAY
        self.total_agents:int       = args.agents_number
        self.batch_size: int        = args.batch_size
        self.epsilon_decay: float   = args.epsilon_decay
        self.mask:bool              = args.mask
        self.neighbor_distance:int  = args.neighbor_distance
        self.epsilon: float         = args.epsilon if hasattr(args, 'epsilon') else MAX_EPSILON
        self.device: torch.device   = torch.device(args.device)

    @abstractmethod
    def get_action(self, global_state: Tuple[PositionType, ...]) -> int:
        pass
    
    def decay_epsilon_power(self, step: int):
        """
        ステップ数に基づき、探索率εを指数的に減衰させる関数。
        Args:
            step (int): 現在のステップ数（またはエピソード数）。
        """
        lambda_ = 0.0001
        # 指数減衰式: ε_t = ε_start * (decay_rate)^t
        # self.epsilon = MAX_EPSILON * (self.epsilon_decay ** (step*lambda_))
        self.epsilon *= MAX_EPSILON * (self.epsilon_decay ** (lambda_))

        # 最小値（例: 0.01）を下回らないようにすることが多いが、ここではシンプルな式のみを返します。
        self.epsilon = max(MIN_EPSILON, self.epsilon)
        
    @abstractmethod
    def get_all_q_values(self, global_state: Tuple[PositionType, ...]) -> Any:
        pass

    @abstractmethod
    def observe(self, global_state: Tuple[PositionType, ...], action: int, reward: float, next_global_state: Tuple[PositionType, ...], done: bool) -> None:
        pass

    @abstractmethod
    def learn(self, total_step: int | None = None) -> float | None:
        pass

    @abstractmethod
    def set_weights(self, weights: Any) -> None:
        pass

    @abstractmethod
    def get_weights(self) -> Any:
        pass

