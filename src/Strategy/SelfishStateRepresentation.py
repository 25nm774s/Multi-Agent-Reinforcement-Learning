from typing import List, Tuple
from Strategy.StateRepresentationStrategy import StateRepresentationStrategy
from Base.Constant import GlobalState

class SelfishStateRepresentation(StateRepresentationStrategy):
    """利己的戦略の状態表現（自身のエージェントの位置 + すべての目標の位置）を生成する具体的な戦略。"""

    UNOBSERVED_POSITION = (-1, -1)

    def __init__(self, grid_size: int, goals_num: int, agent_id: int, total_agents: int):
        self.grid_size = grid_size
        self.goals_num = goals_num
        self.agent_id = agent_id
        self.total_agents = total_agents

    def get_q_state_representation(self, global_state: GlobalState, _=None) -> GlobalState:
        expected_len = self.goals_num + self.total_agents
        if len(global_state) != expected_len:
            raise ValueError(f"サイズ不一致: 期待 {expected_len}, 実際 {len(global_state)}")

        # 1. ゴール情報の抽出
        goal_positions = list(global_state[:self.goals_num])
        # リスト作成時に型を明示する
        agent_info: List[Tuple[int, int]] = [self.UNOBSERVED_POSITION] * self.total_agents
        # これで、自分の位置（任意の整数タプル）を代入できるようになります
        
        # 2. エージェント情報の構築 (自分以外を非観測にする)
        # 自分の位置だけ上書き (global_state 内のインデックスは goals_num + agent_id)
        my_pos = global_state[self.goals_num + self.agent_id]
        
        if not isinstance(my_pos, tuple) or len(my_pos) != 2:
            raise ValueError(f"不正な座標形式: {my_pos}")
             
        agent_info[self.agent_id] = my_pos

        # 3. 新しい状態表現として結合して返す (元の global_state は変更しない)
        return tuple(goal_positions + agent_info)