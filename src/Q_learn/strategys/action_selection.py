import abc
import numpy as np

from ..QTable import QTable,QState

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

class StandardActionSelection(ActionSelectionStrategy):
    """Concrete Strategy for Standard Epsilon-Greedy Action Selection."""

    def select_action(self, q_table: QTable, q_state: QState, action_size: int, epsilon: float) -> int:
        """
        Select an action using standard epsilon-greedy based on the agent's Q-table.
        """
        if np.random.rand() < epsilon:
            # Explore: Choose a random action
            return np.random.choice(action_size)
        else:
            # Exploit: Choose the best action based on Q-values
            # Use the QTable instance to get Q-values for this specific state
            q_values = q_table.get_q_values(q_state)

            # Select the action with the maximum Q-value (break ties randomly)
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)
