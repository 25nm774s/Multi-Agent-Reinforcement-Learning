from abc import ABC, abstractmethod
import numpy as np

# --- 観測空間と行動空間を模倣するクラス ---
# gymnasium.spaces.Box を模倣したもの
# 連続値をとる空間を定義するために使用
class CustomBox:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        """空間内からランダムな値をサンプリングする"""
        return self.low + (self.high - self.low) * np.random.rand(*self.shape)

    def contains(self, x):
        """値が空間内に含まれるかチェックする"""
        return (x >= self.low).all() and (x <= self.high).all()

    def __repr__(self):
        return f"CustomBox({self.low[0]}..{self.high[0]}, shape={self.shape}, dtype={self.dtype})"

# gymnasium.spaces.Discrete を模倣したもの
# 離散値をとる空間を定義するために使用
class CustomDiscrete:
    def __init__(self, n):
        self.n = n # 取りうる離散値の数 (0 から n-1)

    def sample(self):
        """空間内からランダムな値をサンプリングする"""
        return np.random.randint(self.n)

    def contains(self, x):
        """値が空間内に含まれるかチェックする"""
        return 0 <= x < self.n and isinstance(x, (int, np.integer))

    def __repr__(self):
        return f"CustomDiscrete({self.n})"

# gymnasium.spaces.Dict を模倣したもの (エージェントごとの空間を定義するため)
class CustomDict:
    def __init__(self, spaces):
        self.spaces = spaces # 辞書形式で、キーがエージェントID、値が空間オブジェクト

    def sample(self):
        """各エージェントの空間からランダムな値をサンプリングし、辞書で返す"""
        return {k: space.sample() for k, space in self.spaces.items()}

    def contains(self, x):
        """辞書内の全ての値がそれぞれの空間内に含まれるかチェックする"""
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[k].contains(v) for k, v in x.items())

    def __getitem__(self, key):
        return self.spaces[key]

    def __repr__(self):
        return f"CustomDict({self.spaces})"


# --- マルチエージェント環境の抽象基底クラス ---
class MultiAgentEnvABC(ABC):
    """
    GymnasiumのEnvインターフェースを模倣したマルチエージェント環境の抽象基底クラス。
    このクラスを継承して、具体的な環境を実装します。

    観測空間: エージェントが環境から受け取る情報の形式と範囲を定義
    行動空間: エージェントが環境に対して取りうる行動の形式と範囲を定義
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.observation_space = None  # 各エージェントの観測空間をCustomDictで定義
        self.action_space = None       # 各エージェントの行動空間をCustomDictで定義

        # シード値管理のための擬似乱数ジェネレータ
        self.np_random = np.random.default_rng()

    def set_seed(self, seed: int = None):
        """環境のランダム性を制御するためのシードを設定する（オプション）"""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    @abstractmethod
    def reset(self, seed: int = None, options: dict = None) -> tuple:
        """
        環境を初期状態に戻します。

        Args:
            seed (int, optional): 環境のランダム性を制御するためのシード。
                                  Noneの場合、内部のシードがそのまま使われるか、
                                  ランダムに設定される。
            options (dict, optional): 環境の初期化に関する追加オプション。

        Returns:
            observation (dict): 各エージェントの初期観測を含む辞書。
                                例: {"agent_0": obs_0, "agent_1": obs_1, ...}
            info (dict): 初期状態に関する追加情報を含む辞書。
        """
        if seed is not None:
            self.set_seed(seed)
        pass # 実装は継承クラスで行う

    @abstractmethod
    def step(self, actions: dict) -> tuple:
        """
        エージェントの行動を受け取り、環境を1ステップ進めます。

        Args:
            actions (dict): 各エージェントの行動を含む辞書。
                            例: {"agent_0": action_0, "agent_1": action_1, ...}

        Returns:
            observation (dict): 各エージェントの次の観測を含む辞書。
            rewards (dict): 各エージェントの報酬を含む辞書。
            terminations (dict): 各エージェントが「終了」したかどうかを示すブール値の辞書。
                                 `"__all__": True` は全てのエージェントのエピソードが終了したことを示す。 ex. ゴール条件達成
            truncations (dict): 各エージェントが「打ち切り」になったかどうかを示すブール値の辞書。
                                `__all__": True` は全てのエージェントのエピソードが打ち切りになったことを示す。ex. 最大ステップ数に到達した
            info (dict): 現在のステップに関する追加情報を含む辞書。
        """
        pass # 実装は継承クラスで行う

    @abstractmethod
    def render(self, mode: str = 'human'):
        """
        環境を可視化します（オプション）。

        Args:
            mode (str): レンダリングモード。例: 'human' :ウィンドウ表示(matplotlibに頼る)
                        'rgb_array' : 配列を返す
                        'ansi' : 環境の状態をテキスト文字列として返す
        """
        pass # 実装は継承クラスで行う

    @abstractmethod
    def close(self):
        """
        環境のリソースを解放します（オプション）。
        レンダリングウィンドウのクローズや、ファイルのクローズなど。
        """
        pass # 実装は継承クラスで行う

    # --- ヘルパーメソッド（必要に応じて追加） ---
    def _get_obs(self) -> dict:
        """現在の観測を各エージェントの辞書として返すヘルパーメソッド。
        具体的な環境で実装する必要があるが、抽象メソッドにはしない。"""
        raise NotImplementedError("'_get_obs' method must be implemented by the subclass.")

    def _get_info(self) -> dict:
        """現在の状態に関する追加情報を返すヘルパーメソッド。
        具体的な環境で実装する必要があるが、抽象メソッドにはしない。"""
        return {} # デフォルトでは空の辞書を返す