from typing import List, Tuple
from Strategy.StateRepresentationStrategy import StateRepresentationStrategy
from Base.Constant import GlobalState

class CooperativeStateRepresentation(StateRepresentationStrategy):
    """
    協調戦略の状態表現（すべての目標位置 + 近傍の他エージェント位置）を生成する具体的な戦略。
    """
    UNOBSERVED_POSITION = (-1, -1)

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def get_q_state_representation(self, global_state: GlobalState, neighbor_distance: int = 2) -> GlobalState:
        # 1. 自分の位置を取得
        my_pos_idx = self.goals_num + self.agent_id
        my_pos = global_state[my_pos_idx]

        # 2. ゴール情報はそのままコピー
        new_state: List[Tuple[int, int]] = list(global_state[:self.goals_num])

        # 3. エージェント情報の処理
        for i in range(self.total_agents):
            current_idx = self.goals_num + i
            other_pos = global_state[current_idx]

            if i == self.agent_id:
                # 自分の位置はそのまま追加
                new_state.append(other_pos)
            else:
                # 他人の場合、距離を計算
                d = max(abs(other_pos[0] - my_pos[0]), abs(other_pos[1] - my_pos[1]))
                if d <= neighbor_distance:
                    new_state.append(other_pos)
                else:
                    new_state.append(self.UNOBSERVED_POSITION)

        return tuple(new_state)