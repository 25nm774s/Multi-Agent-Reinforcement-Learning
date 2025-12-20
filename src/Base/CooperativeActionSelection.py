from typing import Tuple, Any
from abc import ABC, abstractmethod

QState = Tuple[int,...]

class QTableActionSelectionStrategy(ABC):
    """Abstract Base Class for QTable-based Action Selection Strategies."""

    @abstractmethod
    def select_action(self, q_table: Any, q_state: QState, action_size: int, epsilon: float) -> int:
        pass
