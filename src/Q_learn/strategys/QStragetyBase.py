from typing import Tuple, List, Any
from abc import ABC, abstractmethod

# --- Type Aliases ---
# 例: (goal1_x, goal1_y, ..., goalG_x, goalG_y, agent_i_x, agent_i_y, ..., agent_N_x, agent_N_y)
PositionType = Tuple[int, int]
QState = Tuple[int, ...] # Qテーブルの状態表現はタプルで表現される
QTableType = dict[QState, List[float]] # QTableクラスの実際の戻り値型
# QTableType はQTableクラスの実際の戻り値型と一致させる

class QTableLearningStrategy(ABC):
    """Abstract Base Class for QTable-based Learning Strategies."""

    @abstractmethod
    def update_q_value(self, q_table: Any, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        pass

class QTableActionSelectionStrategy(ABC):
    """Abstract Base Class for QTable-based Action Selection Strategies."""

    @abstractmethod
    def select_action(self, q_table: Any, q_state: QState, action_size: int, epsilon: float) -> int:
        pass

