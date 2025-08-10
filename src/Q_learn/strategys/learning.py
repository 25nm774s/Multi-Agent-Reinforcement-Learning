import abc

from ..QTable import QTable,QState



class LearningStrategy(abc.ABC):
    """Abstract Base Class for Learning Strategies."""

    @abc.abstractmethod
    def update_q_value(self, q_table: QTable, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        """
        Update the Q-value based on a single experience using the strategy.

        Args:
            q_table (QTable): The QTable instance to update.
            state (QState): The current state.
            action (int): The action taken.
            reward (float): The received reward.
            next_state (QState): The next state.
            done (bool): Flag indicating if the episode is finished.

        Returns:
            float: The absolute TD error of the update.
        """
        pass

class SelfishQLearning(LearningStrategy):
    """Concrete Strategy for Selfish/Independent Q-Learning Update (mask=1)."""

    def update_q_value(self, q_table: QTable, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        """
        Perform a standard Q-learning update on the agent's Q-table,
        ignoring other agents' positions (as they are not in state/next_state for this strategy).
        """
        # QTable.learn already implements the standard Q-learning update logic.
        # This strategy simply delegates the learning responsibility to the QTable instance.
        return q_table.learn(state, action, reward, next_state, done)

#print("Abstract base classes and standard concrete strategies designed.")