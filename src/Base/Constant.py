from typing import Tuple,List,Any,Dict

# 環境から受け取る座標はこっちを使用　共通の座標表現形式
PosType = Tuple[int,int]
"""共通の座標表現形式"""

GlobalState = Dict[str,PosType]
"""共通の状態表現s"""

#####################################
# Qテーブル内部の座標表現形式
QState = Tuple[int, ...]
"""Qテーブル入力用の座標表現形式(旧QState:)"""

QValues = List[float]
"""行動に対するQ値"""

QTableType = Dict[QState, QValues]
"""Qテーブルの構造"""

####################################
# DQN内部の座標表現形式
# 未定義