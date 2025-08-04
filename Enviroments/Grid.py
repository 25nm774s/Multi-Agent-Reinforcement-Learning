class Grid:
    """
    グリッド空間とグリッド上のオブジェクト位置を管理するクラス。
    """
    def __init__(self, grid_size: int):
        """
        指定されたサイズのグリッドを初期化します。

        Args:
            grid_size (int): 正方形グリッドのサイズ (grid_size x grid_size)。
        """
        self.grid_size: int = grid_size
        self._object_positions: dict[str, tuple[int, int]] = {} # obj_id -> (x, y)


    def add_object(self, obj_id: str, position: tuple[int, int]):
        """
        指定された位置にオブジェクトをグリッドに追加します。

        Args:
            obj_id (str): オブジェクトの一意な識別子 (例: 'agent_0', 'goal_1')。
            position (tuple[int, int]): オブジェクトの (x, y) 座標。

        Raises:
            ValueError: 位置がグリッド範囲外の場合、または obj_id が既に存在する場合。
        """
        if not self.is_valid_position(position):
            raise ValueError(f"位置 {position} はグリッド範囲 ({self.grid_size}x{self.grid_size}) 外です。")
        if obj_id in self._object_positions:
            raise ValueError(f"ID '{obj_id}' を持つオブジェクトは既に存在します。")

        self._object_positions[obj_id] = position

    def get_object_position(self, obj_id: str) -> tuple[int, int]:
        """
        オブジェクトの現在の位置を取得します。

        Args:
            obj_id (str): オブジェクトの一意な識別子。

        Returns:
            tuple[int, int]: オブジェクトの (x, y) 座標。

        Raises:
            KeyError: オブジェクトIDが存在しない場合。
        """
        if obj_id not in self._object_positions:
            raise KeyError(f"ID '{obj_id}' を持つオブジェクトは見つかりませんでした。")
        return self._object_positions[obj_id]

    def get_all_object_positions(self) -> dict[str, tuple[int, int]]:
        """
        グリッド内の全てのオブジェクトの位置を取得します。

        Returns:
            dict[str, tuple[int, int]]: オブジェクトIDをその位置にマッピングする辞書。
        """
        return self._object_positions.copy() # 外部からの変更を防ぐためにコピーを返します。

    def is_position_occupied(self, position: tuple[int, int], exclude_obj_id: str = None) -> bool:
        """
        指定された位置が何らかのオブジェクトによって占有されているかチェックします。

        Args:
            position (tuple[int, int]): チェックする (x, y) 位置。
            exclude_obj_id (str, optional): チェックから除外するオブジェクトID (例: オブジェクト自身)。デフォルトは None。

        Returns:
            bool: 位置が占有されていれば True、そうでなければ False。
        """
        for obj_id, pos in self._object_positions.items():
            if pos == position and (exclude_obj_id is None or obj_id != exclude_obj_id):
                return True
        return False

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        """
        指定された位置がグリッド範囲内にあるかチェックします。

        Args:
            position (tuple[int, int]): チェックする (x, y) 位置。

        Returns:
            bool: 位置が有効であれば True、そうでなければ False。
        """
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    def calculate_next_position(self, current_pos: tuple[int, int], action: int) -> tuple[int, int]:
        """
        現在の位置と行動に基づいて潜在的な次の位置を計算します。
        境界チェックや衝突検出は**行いません**。

        Args:
            current_pos (tuple[int, int]): 現在の (x, y) 位置。
            action (int): 取られた行動 (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY)。

        Returns:
            tuple[int, int]: 潜在的な次の (x, y) 位置。
        """
        x, y = current_pos
        if action == 0: # UP (上方向はy座標が減少)
            y -= 1
        elif action == 1: # DOWN (下方向はy座標が増加)
            y += 1
        elif action == 2: # LEFT (左方向はx座標が減少)
            x -= 1
        elif action == 3: # RIGHT (右方向はx座標が増加)
            x += 1
        # action == 4 (STAY) の場合、(x, y) はそのままになります。

        return (x, y)

    def set_object_position(self, obj_id: str, position: tuple[int, int]):
        """
        既存のオブジェクトの位置を直接設定します。
        衝突検出は**行いません**。境界チェックのみ行います。

        Args:
            obj_id (str): 位置を設定するオブジェクトのID。
            position (tuple[int, int]): 新しい (x, y) 位置。

        Raises:
            KeyError: オブジェクトIDが存在しない場合。
            ValueError: 位置がグリッド範囲外の場合。
        """
        if obj_id not in self._object_positions:
            raise KeyError(f"ID '{obj_id}' を持つオブジェクトは見つかりませんでした。")
        if not self.is_valid_position(position):
            raise ValueError(f"位置 {position} はグリッド範囲外です。")

        self._object_positions[obj_id] = position

    def resolve_agent_movements(self, agent_actions: dict[str, int]) -> dict[str, tuple[int, int]]:
        """
        エージェントの行動を受け取り、衝突解決後の最終的な位置を計算します。
        エージェントの位置を**更新**します。

        Args:
            agent_actions (dict[str, int]): エージェントIDをキー、行動(int)を値とする辞書。

        Returns:
            dict[str, tuple[int, int]]: エージェントIDをキー、衝突解決後の最終位置を値とする辞書。
        """
        potential_positions: dict[str, tuple[int, int]] = {}
        current_agent_positions: dict[str, tuple[int, int]] = {}

        # 現在のエージェント位置を取得し、潜在的な次の位置を計算します
        agent_ids = list(agent_actions.keys()) # 行動が指定されたエージェントのみを考慮
        for agent_id in agent_ids:
            if agent_id not in self._object_positions:
                # Gridに存在しないエージェントが行動を試みた場合
                raise KeyError(f"Agent with ID '{agent_id}' not found in grid.")

            current_pos = self._object_positions[agent_id]
            current_agent_positions[agent_id] = current_pos
            potential_positions[agent_id] = self.calculate_next_position(current_pos, agent_actions[agent_id])

        # 衝突解決後の最終的な位置を決定します
        final_positions: dict[str, tuple[int, int]] = {}

        # エージェントごとに最終位置を決定します
        for agent_id in agent_ids:
            potential_pos = potential_positions[agent_id]
            current_pos = current_agent_positions[agent_id]

            # 境界チェック
            if not self.is_valid_position(potential_pos):
                final_positions[agent_id] = current_pos # 境界に当たった場合は静止
                continue

            # 他のエージェントとの衝突をチェックします
            collision_detected = False
            for other_agent_id in agent_ids:
                if agent_id == other_agent_id:
                    continue

                # 自分の潜在的な位置が、他のエージェントの意図する位置と衝突するか
                if potential_pos == potential_positions[other_agent_id]:
                    collision_detected = True
                    break

                # 自分の潜在的な位置が、他のエージェントの現在の位置と衝突するか (他のエージェントが静止または自分に向かってくる場合など)
                # シンプルなルールとして、移動先が他のエージェントの「現在の」位置と一致する場合も衝突とみなす
                if potential_pos == current_agent_positions[other_agent_id]:
                    collision_detected = True
                    break

            # ゴール位置との衝突は許容されるため、ここではチェックしません。
            # 障害物（もしあれば）との衝突は is_position_occupied などを使ってここでチェックできますが、
            # 現状はエージェントとゴールのみなので、エージェント間衝突のみ考慮します。


            if collision_detected:
                final_positions[agent_id] = current_pos # 衝突が検出された場合は静止
            else:
                final_positions[agent_id] = potential_pos # 潜在的な位置に移動


        # グリッド内のエージェント位置を更新します
        for agent_id, new_pos in final_positions.items():
            self.set_object_position(agent_id, new_pos) # Grid内で位置を更新


        return final_positions # 衝突解決後の最終位置を返します
