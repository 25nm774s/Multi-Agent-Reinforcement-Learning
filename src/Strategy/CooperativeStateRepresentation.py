from typing import Any, List, Tuple
from Strategy.StateRepresentationStrategy import StateRepresentationStrategy, PositionType, QState

class CooperativeStateRepresentation(StateRepresentationStrategy):
    """
    協調戦略の状態表現（すべての目標位置 + 他エージェント位置）を生成する具体的な戦略。
    """
    UNOBSERVED_POSITION = (-1, -1)

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def get_q_state_representation(self, global_state: Tuple[PositionType, ...], neighbor_distance: int) -> QState:
        goal_positions = global_state[:self.goals_num]
        agent_positions = global_state[self.goals_num:]
        agent_position = agent_positions[self.agent_id]

        flat_state_list: List[int] = []

        # ゴールはそのまま
        for p in goal_positions:
            flat_state_list.extend(p)

        for i, pos in enumerate(agent_positions):
            if i == self.agent_id:
                flat_state_list.extend(pos)
                continue    # 自身を除く

            d = max(abs(pos[0] - agent_position[0]), abs(pos[1] - agent_position[1]))

            if d <= neighbor_distance:
                flat_state_list.extend(pos)
            else:
                flat_state_list.extend(self.UNOBSERVED_POSITION)

        return tuple(flat_state_list)
        