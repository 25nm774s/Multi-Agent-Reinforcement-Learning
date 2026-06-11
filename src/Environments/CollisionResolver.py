from .Grid import Grid
from Base.Constant import PosType

class CollisionResolver:
    """
    グリッド上でのオブジェクト（特にエージェント）の移動と衝突解決を担当するクラス。
    Grid クラスのインスタンスを参照して位置情報にアクセスします。
    """
    def __init__(self, grid: Grid):
        """
        CollisionResolver を初期化します。

        Args:
            grid (Grid): 位置情報にアクセスするための Grid クラスのインスタンス。
        """
        self._grid: Grid = grid
        self.action_space_size = 5 # 行動空間をここで定義

    def _calculate_next_position(self, current_pos: PosType, action: int) -> PosType:
        """
        現在の位置と行動に基づいて潜在的な次の位置を計算します。
        境界チェックや衝突検出は**行いません**。

        Args:
            current_pos (PosType): 現在の (x, y) 位置。
            action (int): 取られた行動 (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY)。

        Returns:
            PosType: 潜在的な次の (x, y) 位置。
        """
        x, y = current_pos
        if action == 0: # UP (上方向はy座標が減少)
            y -= 1
        elif action == 1: # DOWN (下方向はyが増加)
            y += 1
        elif action == 2: # LEFT (左方向はxが減少)
            x -= 1
        elif action == 3: # RIGHT (右方向はxが増加)
            x += 1
        # action == 4 (STAY) の場合、(x, y) はそのままになります。

        return (x, y)

    def resolve_agent_movements(self, agent_actions: dict[str, int]) -> dict[str, PosType]:
        """
        エージェントの行動を受け取り、衝突解決後の最終的な位置を計算します。
        参照している Grid インスタンスのエージェント位置を**更新**します。

        Args:
            agent_actions (dict[str, int]): エージェントIDをキー、行動(int)を値とする辞書。

        Returns:
            dict[str, PosType]: エージェントIDをキー、衝突解決後の最終位置を値とする辞書。
        """
        potential_positions: dict[str, PosType] = {}
        current_agent_positions: dict[str, PosType] = {}

        # 現在のエージェント位置を取得し、潜在的な次の位置を計算します
        agent_ids = list(agent_actions.keys()) # 行動が指定されたエージェントのみを考慮
        for agent_id in agent_ids:
            try:
                current_pos = self._grid.get_object_position(agent_id)
                current_agent_positions[agent_id] = current_pos
                potential_positions[agent_id] = self._calculate_next_position(current_pos, agent_actions[agent_id])
            except KeyError:
                # Gridに存在しないエージェントが行動を試みた場合
                raise KeyError(f"Agent with ID '{agent_id}' not found in grid.")


        # 衝突解決後の最終的な位置を決定します
        final_positions: dict[str, PosType] = {}

        # エージェントごとに最終位置を決定します
        for agent_id in agent_ids:
            potential_pos = potential_positions[agent_id]
            current_pos = current_agent_positions[agent_id]

            # 境界チェック
            if not self._grid.is_valid_position(potential_pos):
                final_positions[agent_id] = current_pos # 境界に当たった場合は静止
            else:
                final_positions[agent_id] = potential_pos # 衝突を許容して潜在的な位置に移動

        # Grid内のエージェント位置を更新します
        for agent_id, new_pos in final_positions.items():
            # set_object_position 内で境界チェックは行われるが、ここでは既に有効な位置なので問題ない
            try:
                self._grid.set_object_position(agent_id, new_pos)
            except KeyError:
                # ここで KeyError が発生することはないはずだが、念のため
                print(f"Warning: Agent ID '{agent_id}' not found in grid during position update.")
            except ValueError:
                # ここで ValueError が発生することはないはずだが、念のため
                print(f"Warning: Invalid position '{new_pos}' for Agent ID '{agent_id}' during position update.")


        return final_positions # 衝突解決後の最終位置を返します
