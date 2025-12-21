from abc import ABC, abstractmethod
from typing import Tuple

from Base.Constant import GlobalState

class StateRepresentationStrategy(ABC):
    """Abstract Base Class for State Representation Strategies."""

    UNOBSERVED_POSITION = (-1, -1)

    @abstractmethod
    def get_q_state_representation(self, global_state: GlobalState, neighbor_distance: int) -> GlobalState:
        """
        この戦略のQテーブル状態表現を、グローバル環境状態から生成します。

        Args:
            global_state (Tuple[PositionType, ...]): The global state tuple
                                                        ((goal1_x, goal1_y), ..., (agent1_x, agent1_y), ...).
            neighbor_distance (int): 観測空間の有効距離

        Returns:
            QState: The state representation suitable for the QTable lookup for this strategy.
        """
        pass

