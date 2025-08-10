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

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        """
        Initializes the SelfishQLearning strategy.
        Added __init__ to accept necessary parameters.
        """
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def update_q_value(self, q_table: QTable, state: QState, action: int, reward: float, next_state: QState, done: bool) -> float:
        """
        Perform a standard Q-learning update on the agent's Q-table.
        """
        return q_table.learn(state, action, reward, next_state, done)
