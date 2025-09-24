from abc import ABC, abstractmethod
import numpy as np

from ..QTable import QTable, QState

from typing import Tuple, List

class ActionSelectionStrategy(ABC):
    """Abstract Base Class for Action Selection Strategies."""

    @abstractmethod
    def select_action(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        """
        戦略に基づいて、現在の状態とQ値に基づいて行動を選択します。

        Args:
            q_table (QTable): The QTable instance containing Q-values.
            q_state (QState): The agent's current state representation.
            action_size (int): The size of the action space.
            epsilon (float): The current epsilon value for exploration.

        Returns:
            int: The selected action.
        """
        pass

    @abstractmethod
    def get_q_state_representation(self, global_state: Tuple[Tuple[int, int], ...], neighbor_distance:int) -> QState:
        """
        この戦略のQテーブル状態表現を、グローバル環境状態から生成します。

        Args:
            global_state (Tuple[Tuple[int, int], ...]): The global state tuple
                                                        ((goal1_x, goal1_y), ..., (agent1_x, agent1_y), ...).
            neighbor_distance (int): 観測空間の有効距離

        Returns:
            QState: The state representation suitable for the QTable lookup for this strategy.
        """
        pass

class SelfishActionSelection(ActionSelectionStrategy):
    """利己的/独立したエプシロン・グリーディ行動選択のための具体的な戦略 (mask=1)."""

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        """
        Initializes the SelfishActionSelection strategy.
        """
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def select_action(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        """
        エージェントのQテーブルに基づいて、標準的なε-グリーディー法を使用してアクションを選択します。
        """
        if np.random.rand() < epsilon:
            return np.random.choice(action_size)
        else:
            q_values = q_table.get_q_values(q_state)
            if not q_values:
                return np.random.choice(action_size)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def get_q_state_representation(self, global_state: Tuple[Tuple[int, int], ...], _) -> QState:
        """
        利己的戦略の状態で表す（自身のエージェントの位置 + すべての目標の位置）。
        """
        expected_len = self.goals_num + self.total_agents
        if len(global_state) != expected_len:
            raise ValueError(f"SelfishActionSelection におけるグローバル状態のサイズが不一致。期待値は{expected_len}ですが、実際は{len(global_state)}でした。")

        goal_positions = global_state[:self.goals_num]
        agent_position = global_state[self.goals_num + self.agent_id]

        flat_state_list: List[int] = []
        for pos in goal_positions:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"予期しないゴール位置のフォーマット: {pos} in SelfishActionSelection.get_q_state_representation")
            flat_state_list.extend(pos)

        if not isinstance(agent_position, tuple) or len(agent_position) != 2:
            raise ValueError(f"予期しないエージェント位置形式: {agent_position} in SelfishActionSelection.get_q_state_representation")
        flat_state_list.extend(agent_position)

        return tuple(flat_state_list)
