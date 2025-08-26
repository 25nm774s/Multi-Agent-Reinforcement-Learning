from typing import Tuple

PositionType = Tuple[int,int]

class Grid:
    """
    グリッド空間とグリッド上のオブジェクト位置を管理するクラス。
    純粋な座標管理機能を提供します。
    """
    def __init__(self, grid_size: int):
        """
        指定されたサイズのグリッドを初期化します。

        Args:
            grid_size (int): 正方形グリッドのサイズ (grid_size x grid_size)。
        """
        self.grid_size: int = grid_size
        self._object_positions: dict[str, PositionType] = {} # obj_id -> (x, y)

    def add_object(self, obj_id: str, position: PositionType):
        """
        オブジェクトを指定されたグリッド位置に追加します。

        Args:
            obj_id (str): オブジェクトの一意な識別子 (例: 'agent_0', 'goal_1')。
            position (PositionType): オブジェクトの (x, y) 座標。

        Raises:
            ValueError: 位置がグリッド範囲外の場合、または obj_id が既に存在する場合。
        """
        if not self.is_valid_position(position):
            raise ValueError(f"位置 {position} はグリッド範囲 ({self.grid_size}x{self.grid_size}) 外です。")
        if obj_id in self._object_positions:
            raise ValueError(f"ID '{obj_id}' を持つオブジェクトは既に存在します。")

        self._object_positions[obj_id] = position

    def get_object_position(self, obj_id: str) -> PositionType:
        """
        オブジェクトの現在の位置を取得します。

        Args:
            obj_id (str): オブジェクトの一意な識別子。

        Returns:
            PositionType: オブジェクトの (x, y) 座標。

        Raises:
            KeyError: オブジェクトIDが存在しない場合。
        """
        if obj_id not in self._object_positions:
            raise KeyError(f"ID '{obj_id}' を持つオブジェクトは見つかりませんでした。")
        return self._object_positions[obj_id]

    def get_all_object_positions(self) -> dict[str, PositionType]:
        """
        グリッド内の全てのオブジェクトの位置を取得します。

        Returns:
            dict[str, PositionType]: オブジェクトIDをその位置にマッピングする辞書。
        """
        return self._object_positions.copy() # 外部からの変更を防ぐためにコピーを返します。

    def is_position_occupied(self, position: PositionType, exclude_obj_id: str = None) -> bool:# type:ignore
        """
        指定された位置が何らかのオブジェクトによって占有されているかチェックします。

        Args:
            position (PositionType): チェックする (x, y) 位置。
            exclude_obj_id (str, optional): チェックから除外するオブジェクトID (例: オブジェクト自身)。デフォルトは None。

        Returns:
            bool: 位置が占有されていれば True、そうでなければ False。
        """
        for obj_id, pos in self._object_positions.items():
            if pos == position and (exclude_obj_id is None or obj_id != exclude_obj_id):
                return True
        return False

    def is_valid_position(self, position: PositionType) -> bool:
        """
        指定された位置がグリッド範囲内にあるかチェックします。

        Args:
            position (PositionType): チェックする (x, y) 位置。

        Returns:
            bool: 位置が有効であれば True、そうでなければ False。
        """
        return 0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size

    def set_object_position(self, obj_id: str, position: PositionType):
        """
        既存のオブジェクトの位置を直接設定します。
        衝突検出は**行いません**。境界チェックのみ行います。

        Args:
            obj_id (str): 位置を設定するオブジェクトのID。
            position (PositionType): 新しい (x, y) 位置。

        Raises:
            KeyError: オブジェクトIDが存在しない場合。
            ValueError: 位置がグリッド範囲外の場合。
        """
        if obj_id not in self._object_positions:
            raise KeyError(f"ID '{obj_id}' を持つオブジェクトは見つかりませんでした。")
        if not self.is_valid_position(position):
            raise ValueError(f"位置 {position} はグリッド範囲外です。")

        self._object_positions[obj_id] = position

    def remove_object(self, obj_id: str):
        """
        グリッドからオブジェクトを削除します。

        Args:
            obj_id (str): 削除するオブジェクトのID。

        Raises:
            KeyError: オブジェクトIDが存在しない場合。
        """
        if obj_id not in self._object_positions:
            raise KeyError(f"ID '{obj_id}' を持つオブジェクトは見つかりませんでした。")
        del self._object_positions[obj_id]
