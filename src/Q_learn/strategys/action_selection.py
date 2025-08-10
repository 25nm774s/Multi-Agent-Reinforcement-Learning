import abc
import numpy as np

from ..QTable import QTable,QState

from typing import Tuple, List

class ActionSelectionStrategy(abc.ABC):
    """Abstract Base Class for Action Selection Strategies."""

    @abc.abstractmethod
    def select_action(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        """
        Select an action based on the current state and Q-values using the strategy.

        Args:
            q_table (QTable): The QTable instance containing Q-values.
            q_state (QState): The agent's current state representation.
            action_size (int): The size of the action space.
            epsilon (float): The current epsilon value for exploration.

        Returns:
            int: The selected action.
        """
        pass

    @abc.abstractmethod
    def get_q_state_representation(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
        """
        Generate the Q-table state representation for this strategy from the global environment state.

        Args:
            global_state (Tuple[Tuple[int, int], ...]): The global state tuple
                                                        ((goal1_x, goal1_y), ..., (agent1_x, agent1_y), ...).

        Returns:
            QState: The state representation suitable for the QTable lookup for this strategy.
        """
        pass

class SelfishActionSelection(ActionSelectionStrategy):
    """Concrete Strategy for Selfish/Independent Epsilon-Greedy Action Selection (mask=1)."""

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
        Select an action using standard epsilon-greedy based on the agent's Q-table.
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

    def get_q_state_representation(self, global_state: Tuple[Tuple[int, int], ...]) -> QState:
        """
        Generate the state representation for the Selfish strategy (own agent position + all goal positions).
        """
        expected_len = self.goals_num + self.total_agents
        if len(global_state) != expected_len:
            raise ValueError(f"Global state length mismatch in SelfishActionSelection. Expected {expected_len}, got {len(global_state)}")

        goal_positions = global_state[:self.goals_num]
        agent_position = global_state[self.goals_num + self.agent_id]

        flat_state_list: List[int] = []
        for pos in goal_positions:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"Unexpected goal position format: {pos} in SelfishActionSelection.get_q_state_representation")
            flat_state_list.extend(pos)

        if not isinstance(agent_position, tuple) or len(agent_position) != 2:
            raise ValueError(f"Unexpected agent position format: {agent_position} in SelfishActionSelection.get_q_state_representation")
        flat_state_list.extend(agent_position)

        return tuple(flat_state_list)
