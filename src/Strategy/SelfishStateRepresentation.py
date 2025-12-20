from typing import List, Tuple
from Base.StateRepresentationStrategy import StateRepresentationStrategy, PositionType, QState

class SelfishStateRepresentation(StateRepresentationStrategy):
    """利唳的戦略の状態表現（自身のエージェントの位置 + すべての目標の位置）を生成する具体的な戦略。"""

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def get_q_state_representation(self, global_state: Tuple[PositionType, ...], _) -> QState:
        expected_len = self.goals_num + self.total_agents
        if len(global_state) != expected_len:
            raise ValueError(f"SelfishStateRepresentation におけるグローバル状態のサイズが不一致。期待値は{expected_len}ですが、実際は{len(global_state)}でした。")

        goal_positions = global_state[:self.goals_num]
        agent_position = global_state[self.goals_num + self.agent_id]

        flat_state_list: List[int] = []
        for pos in goal_positions:
            if not isinstance(pos, tuple) or len(pos) != 2:
                raise ValueError(f"予期しないゴール位置のフォーマット: {pos} in SelfishStateRepresentation.get_q_state_representation")
            flat_state_list.extend(pos)

        if not isinstance(agent_position, tuple) or len(agent_position) != 2:
            raise ValueError(f"予期しないエージェント位置形式: {agent_position} in SelfishStateRepresentation.get_q_state_representation")
        flat_state_list.extend(agent_position)

        return tuple(flat_state_list)

