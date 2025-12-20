import numpy as np
from typing import Any
from .QStragetyBase import QTableLearningStrategy, QTableActionSelectionStrategy, QState

# --- 3.2. SelfishActionSelection (QTable): 利己的行動選択戦略 ---
class SelfishStrategy(QTableActionSelectionStrategy, QTableLearningStrategy):
    """利己的/独立したエプシロン・グリーディ行動選択のための具体的な戦略 (mask=1)."""

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def select_action(self, q_table: Any, q_state: QState, action_size: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.choice(action_size)
        else:
            q_values = q_table.get_q_values(q_state)
            if not q_values:
                return np.random.choice(action_size)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)

    def update_q_value(self, q_table: Any, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        return q_table.learn(state, action, reward, next_state, done)
