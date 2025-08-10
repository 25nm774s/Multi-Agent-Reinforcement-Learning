"""
線形関数近似器によるQ学習の実装.
状態数だけガウス分布を生成して基底関数とする.
学習済みパラメータを用いる場合は指定CSVから読み込む.
self.sigmaはガウス基底の標準偏差(バンド幅).
"""

import csv
import numpy as np
import os
from typing import List, Tuple, Any, Optional # 型ヒントのためにtypingモジュールをインポート

class Linear:
    # クラス変数：他エージェントと価値共有する場合に使用
    # Q学習のパラメータ(theta)を保持
    # common_theta_listの型は (other_agents_states_size, action_size, b) の形状を持つnp.ndarray
    common_theta_list: Optional[np.ndarray] = None

    def __init__(self, args: Any, action_size: int):
        """
        Linearクラスのコンストラクタ.
        各種パラメータを設定し、Q値のパラメータを初期化または読み込む.

        Args:
            args: コマンドライン引数などの設定を含むオブジェクト.
            action_size: 行動空間のサイズ.
        """
        self.max_ts: int = args.max_timestep
        self.goals_number: int = args.goals_number
        self.agents_number: int = args.agents_number
        self.gamma: float = args.gamma          # 割引率
        self.lr: float = args.learning_rate     # 学習率
        self.grid_size: int = args.grid_size
        self.action_size: int = action_size
        self.batch_size: int = args.batch_size
        self.load_model: int = args.load_model # 0:未学習, 1:学習済みモデル, 2:（元:真のQ値）現在は廃止の方向
        self.mask: int = args.mask             # マスク処理を行うか
        self.model_path: str = args.model_path # モデル保存・読み込みパス

        # スケール調整用ファクター (RBF計算に使用)
        self.norm_factor: np.ndarray = np.array([1, self.grid_size]).reshape(-1, 1)

        # エピソード中のデータを一時的に保持するリスト
        # episode_dataの要素型は (st_idx, action, agent_pos, reward, next_st_idx, next_agent_pos, done)
        self.loss: List[float] = []
        self.episode_data: List[Tuple[int, int, np.ndarray, float, int, np.ndarray, bool]] = []

        # 状態インデックスの計算結果をキャッシュ
        # state_key (タプルのタプル) -> index (int)
        self.index_cache: dict[Tuple[Tuple[int, int], ...], int] = {}

        # グリッド全セル数
        b: int = self.grid_size ** 2

        # 価値共有用のクラス変数を初期化 (Q学習のパラメータ)
        if Linear.common_theta_list is None:
            # agents_num - 1 が負の値にならないようにチェック
            other_agents_states_size: int = b**(max(0, self.agents_number-1))
            # common_theta_list は (other_agents_states_size, action_size, b) の形状を持つNumPy配列
            Linear.common_theta_list = np.zeros((other_agents_states_size, action_size, b))


        # 基底関数用の中心 (mu) を計算 (グリッドの各セルに対応)
        self.mu_array: np.ndarray = np.zeros((2, b))
        cnt = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.mu_array[0, cnt] = i
                self.mu_array[1, cnt] = j
                cnt += 1
        self.sigma: float = 0.3 # ガウス基底の標準偏差(バンド幅)

        # 行動数分のmuを保持する配列 (現状1つのみだが、拡張性を考慮しリスト形式)
        # mu_list の要素型は (2, b) の形状を持つnp.ndarray
        self.mu_list: List[np.ndarray] = [np.copy(self.mu_array)]

        # モデル読み込みまたは真の価値関数として読み込む場合
        # theta_list の型は (action_size, b) の形状を持つNumPy配列のリスト
        self.theta_list: List[np.ndarray]

        if self.load_model == 1:
            # 学習済みモデルを読み込み
            print(f"load_model {self.load_model}: 学習済みモデルを読み込み")
            self.load_model_params(self.model_path, b)
        elif self.load_model == 2:
             # 真の行動価値関数を読み込み -> このモードは廃止し、load_model=1に統合または別クラスで扱うことを検討
             print(f"load_model {self.load_model}: 真の行動価値関数として読み込み (廃止予定)")
             # 現在はload_model=1と同じ動作としておくか、警告を出すなど対応
             self.load_model_params(self.model_path, b) # とりあえず学習済みモデルとしてロード
             # Q用のtheta_listを(最低限)初期化 (真の価値をターゲットに学習するため - このロジックは別途考慮)
             self.theta_list = [np.zeros(b) for _ in range(self.action_size)]
        else:
            # 未学習の初期化
            print(f"load_model {self.load_model}: 未学習のため初期化")
            self.theta_list = [np.zeros(b) for _ in range(self.action_size)]

    def load_model_params(self, model_path: str, b: int) -> None:
        """
        学習済みのQ関数パラメータをCSVから読み込んでtheta_listに格納する.
        save_modelと対になるメソッド.

        Args:
            model_path: 学習済みモデルのCSVファイルパス.
            b: グリッドの全セル数.
        """
        if not model_path: # パスが空の場合は処理をスキップ
             print("モデルファイルパスが指定されていません。")
             self.theta_list = [np.zeros(b) for _ in range(self.action_size)]
             return

        if not os.path.exists(model_path):
            print(f"指定されたモデルファイルが見つかりません: {model_path}")
            # ファイルが見つからない場合は初期化を行うなど、適切なハンドリングを追加
            self.theta_list = [np.zeros(b) for _ in range(self.action_size)]
            return

        try:
            with open(model_path, 'r') as f:
                # CSVリーダーから直接numpy配列のリストを作成
                # rows: List[np.ndarray] の形状 (N, b)
                rows: List[np.ndarray] = [np.array(row, dtype=float) for row in csv.reader(f)]

            # ロードしたデータの形状を確認
            loaded_theta_array = np.array(rows) # (N, b) の形状を持つnp.ndarray
            b = self.grid_size ** 2 # bを再計算

            if self.mask == 0:
                # マスクなし: common_theta_list をロード
                other_agents_states_size: int = b**(max(0, self.agents_number - 1))
                expected_shape: Tuple[int, int] = (other_agents_states_size * self.action_size, b) # 保存時のフラット化された形状

                if loaded_theta_array.shape == expected_shape:
                    # ロードしたデータを common_theta_list の形状に戻す
                    Linear.common_theta_list = loaded_theta_array.reshape(other_agents_states_size, self.action_size, b)
                else:
                     print(f"Warning: ロードしたモデルの形状 {loaded_theta_array.shape} が期待される形状 {expected_shape} と異なります。(マスクなし)")
                     # 形状が異なる場合のハンドリング（例: 初期化し直す、エラーを出すなど）
                     expected_common_shape: Tuple[int, int, int] = (other_agents_states_size, self.action_size, b)
                     Linear.common_theta_list = np.zeros(expected_common_shape) # 安全のため初期化
                     self.theta_list = [np.zeros(b) for _ in range(self.action_size)] # 自エージェント用も初期化


            else:
                # マスクあり: self.theta_list をロード
                expected_shape: Tuple[int, int] = (self.action_size, b)
                if loaded_theta_array.shape == expected_shape:
                    # ロードしたデータを theta_list の形状に戻す (np.ndarrayのリスト)
                    self.theta_list = list(loaded_theta_array) # NumPy配列のリストに変換
                else:
                    print(f"Warning: ロードしたモデルの形状 {loaded_theta_array.shape} が期待される形状 {expected_shape} と異なります。(マスクあり)")
                    self.theta_list = [np.zeros(b) for _ in range(self.action_size)] # 初期化


        except Exception as e:
            print(f"モデルのロード中にエラーが発生しました: {e}")
            # エラー発生時も初期化を行うなど、適切なハンドリングを追加
            b = self.grid_size ** 2 # bを再計算
            self.theta_list = [np.zeros(b) for _ in range(self.action_size)]
            if self.mask == 0:
                 b = self.grid_size ** 2
                 other_agents_states_size = b**(max(0, self.agents_number - 1))
                 expected_common_shape = (other_agents_states_size, self.action_size, b)
                 Linear.common_theta_list = np.zeros(expected_common_shape)


    def save_model_params(self, file_path: str) -> None:
        """
        学習済みのQ関数パラメータ (theta_list または common_theta_list) をCSVに保存する.
        QTableクラスの save_q_table と対になるメソッド.

        Args:
            file_path (str): 保存先のCSVファイルパス.
        """
        save_dir: str = os.path.dirname(file_path)
        if save_dir: # ディレクトリ部分が空でない場合にのみ作成
            os.makedirs(save_dir, exist_ok=True)
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if self.mask:
                    # マスクあり: self.theta_list を保存
                    # self.theta_list は List[np.ndarray]
                    for row in self.theta_list:
                        writer.writerow(row.tolist()) # numpy.ndarray をリストに変換して書き出し
                else:
                    # マスクなし: Linear.common_theta_list を保存 (フラット化して保存)
                    # common_theta_list は (other_agents_states_size, action_size, b) の形状
                    # これを (other_agents_states_size * action_size, b) の形状にフラット化して保存
                    # 各行が b 次元のベクトルになるように書き出す
                    b: int = self.grid_size ** 2
                    other_agents_states_size: int = b**(max(0, self.agents_number - 1))
                    # Linear.common_theta_list が期待される形状であることを確認
                    if Linear.common_theta_list is not None and Linear.common_theta_list.shape == (other_agents_states_size, self.action_size, b):
                        # フラット化して行ごとに書き出し
                        flat_theta: np.ndarray = Linear.common_theta_list.reshape(-1, b)
                        for row in flat_theta:
                            writer.writerow(row.tolist()) # numpy.ndarray をリストに変換して書き出し
                    else:
                        print("Warning: common_theta_list の形状が不正です。保存をスキップしました。")


            print(f"モデルパラメータを {file_path} に保存しました.")
        except Exception as e:
            print(f"モデルパラメータの保存中にエラーが発生しました: {e}")


    def generate_states(self) -> List[List[int]]:
        """
        全エージェント分，セル数を考慮した状態(1次元座標のリスト)を再帰的に生成する.

        Returns:
            全エージェントの全状態の組み合わせリスト.
        """
        states: List[List[int]] = []
        def _dfs(current: List[int], depth: int) -> None:
            if depth == 0:
                states.append(current)
                return
            for i in range(self.grid_size):
                _dfs(current + [i], depth - 1)
        # 各エージェントの状態 (0 to grid_size-1) を組み合わせる
        _dfs([], self.agents_number)
        return states

    def generate_my_states(self) -> List[List[int]]:
        """
        自エージェントのみの全2次元座標を生成する.

        Returns:
            自エージェントの全可能な2次元座標リスト.
        """
        return [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)]

    def rbfs(self, state: np.ndarray) -> np.ndarray:
        """
        RBF (ガウス基底関数) を計算して返す.
        入力状態と各基底中心(mu)との距離に基づき、ガウス関数値を計算する.

        Args:
            state: 現在の自エージェントの2次元座標. (np.ndarray)

        Returns:
            各基底関数に対応する値のnumpy配列. (np.ndarray)
        """
        state = state.reshape(len(state), 1)
        # 距離(L2ノルム)を計算 (norm_factorでスケール調整)
        # mu_list shape: (1, 2, b) or (action_size, 2, b)
        # state shape: (2, 1)
        # Broadcasting will handle (1, 2, b) - (2, 1) -> (1, 2, b)
        # Then norm over axis=1 -> (1, b)
        # Ravel -> (b,)
        dist: np.ndarray = np.linalg.norm((np.array(self.mu_list) - state) / self.norm_factor, axis=1) # mu_listをnp.arrayに変換
        # exp(-d^2/(2σ^2)) を出力
        return np.exp(-dist**2 / (2 * self.sigma**2)).ravel()

    def get_q_values(self, state_tuple: Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]) -> List[float]:
        """
        指定された状態のQ値リストを取得する.
        QTableクラスの get_q_values と対になるメソッド.
        ただし、Linearモデルは状態の組み合わせ全体をキーとするQテーブルを持たず、
        線形近似でQ値を計算するため、このメソッドは擬似的にその役割を果たす。
        入力 state_tuple は (自エージェントのインデックスi, goals_pos, agents_pos) の形式を想定。

        Args:
            state_tuple: (自エージェントのインデックスi, goals_pos, agents_pos) のタプル.

        Returns:
            List[float]: その状態における各行動のQ値リスト.
        """
        i, goals_pos, agents_pos = state_tuple
        agent_pos: np.ndarray = np.array(agents_pos[i]) # 自エージェントの現在の位置 (np.ndarray)

        # 他エージェントの状態リストを作成 (自エージェントの状態を除去)
        # Corrected: Handle the case when agents_number is 1 (no other agents)
        agents_pos_others: List[Tuple[int, int]]
        if self.agents_number > 1:
            agents_pos_others = agents_pos[:i] + agents_pos[i+1:] # 自エージェントの位置を除いたリストを作成
        else: # agents_number == 1
            agents_pos_others = []


        # 各行動に対するQ値を計算してリストとして返す
        q_values: List[float] = [self.getQ(agents_pos_others, agent_pos, a) for a in range(self.action_size)]
        return q_values


    def getQ(self, agents_pos_others: List[Tuple[int, int]], agent_pos: np.ndarray, action: int) -> float:
        """
        自エージェントの(状態, 行動)に対応するQ値を返す.
        線形近似により Q(s,a) = φ(s)・θ_a を計算する.

        Args:
            agents_pos_others: 他エージェントの状態リスト (自エージェントの状態は含まない).
            agent_pos: 自エージェントの2次元座標. (np.ndarray)
            action: 自エージェントの行動.

        Returns:
            計算されたQ値. (float)
        """
        # 自エージェントの状態に対応する基底関数値を計算
        phi_s: np.ndarray = self.rbfs(agent_pos)

        if self.mask:
            # マスクあり: 自エージェントのパラメータのみ使用
            # self.theta_list の要素は np.ndarray であることを想定
            return float(phi_s.dot(self.theta_list[action])) # floatにキャスト
        else:
            # マスクなし: 他エージェントの状態に対応する共有パラメータを使用
            idx: int = self.get_index_from_states(agents_pos_others) # 他エージェントの状態からインデックス取得
            # Linear.common_theta_list は np.ndarray であることを想定
            if Linear.common_theta_list is None:
                 # common_theta_list が None の場合は初期化されていないためエラーとするか、デフォルト値を返すかなどのハンドリングが必要
                 # ここでは初期化されている前提とするが、Noneチェックを追加することも考慮
                 raise ValueError("Linear.common_theta_list is None")
            return float(phi_s.dot(Linear.common_theta_list[idx][action])) # floatにキャスト


    def get_index_from_states(self, states: List[Tuple[int, int]]) -> int:
        """
        複数エージェントの座標(例: [x1,y1],[x2,y2]...) を用いて一意なインデックスを算出する.
        これは、他のエージェントの状態の組み合わせに対応する共有パラメータにアクセスするために使用される.
        計算結果をキャッシュして高速化.

        Args:
            states: 他エージェントの状態のリスト (自エージェントの状態は含まない).

        Returns:
            計算された一意なインデックス. (int)
        """
        # states は [[x1, y1], [x2, y2], ...] の形式で、エージェントi以外のエージェントの位置リスト
        # state_key は タプルのタプル ((x1, y1), (x2, y2), ...)
        state_key: Tuple[Tuple[int, int], ...] = tuple(tuple(pos) for pos in states) #type:ignore リストのリストをタプルのタプルに変換してハッシュ化可能に

        if state_key in self.index_cache:
            return self.index_cache[state_key]

        index: int = 0
        b: int = self.grid_size ** 2 # 基数
        # 各エージェントの状態座標を基にインデックスを計算 (グリッドサイズを基数とする)
        # ここでのstatesは他エージェントの状態リストのみなので、インデックス計算はそのリストの順序に依存する
        for k, (x, y) in enumerate(states):
            # 各エージェントの状態がインデックスに寄与する重みを計算
            # len(states) は他エージェント数
            power: int = b**(len(states) - k - 1)
            # (x*grid_size + y) は2次元座標を1次元に変換したもの (0 to grid_size^2-1)
            index += (x * self.grid_size + y) * power
        self.index_cache[state_key] = index
        return index


    def learn(self, i: int, states: Tuple[Tuple[int, int], ...], action: int, reward: float, next_state: Tuple[Tuple[int, int], ...], done: bool, step: int) -> float:
        """
        単一の経験 (状態, 行動, 報酬, 次の状態, 完了フラグ) に基づいてQ関数パラメータを更新する.
        QTableクラスの learn と対になるメソッド.
        Linearモデルはエピソード終了時に一括更新するため、経験を蓄積する.

        Args:
            i: 自エージェントのインデックス. (int)
            states: (goals + agents) のタプル形式の現在の状態. (Tuple[Tuple[int, int], ...])
            action: 自エージェントの行動. (int)
            reward: 即時報酬. (float)
            next_state: (goals + agents) のタプル形式の次の状態. (Tuple[Tuple[int, int], ...])
            done: エピソード終了フラグ. (bool)
            step: 現在のタイムステップ数. (int)

        Returns:
            計算されたTD誤差 (delta). (エピソード終了時の一括更新では平均誤差を返すなど調整が必要)
        """
        # 状態からゴール位置とエージェント位置を分離
        goals_pos: List[Tuple[int, int]] = list(states[:self.goals_number])
        agents_pos: List[Tuple[int, int]] = list(states[self.goals_number:])
        agent_pos: np.ndarray = np.array(agents_pos[i]) # 自エージェントの現在の位置 (np.ndarray)

        # 次の状態からゴール位置とエージェント位置を分離
        next_goals_pos: List[Tuple[int, int]] = list(next_state[:self.goals_number])
        next_agents_pos: List[Tuple[int, int]] = list(next_state[self.goals_number:])
        next_agent_pos: np.ndarray = np.array(next_agents_pos[i]) # 自エージェントの次の位置 (np.ndarray)

        # 他エージェントの状態リストを作成 (自エージェントの状態を除去)
        # Corrected: Handle the case when agents_number is 1 (no other agents)
        agents_pos_others: List[Tuple[int, int]]
        next_agents_pos_others: List[Tuple[int, int]]
        if self.agents_number > 1:
            agents_pos_others = agents_pos[:i] + agents_pos[i+1:]
            next_agents_pos_others = next_agents_pos[:i] + next_agents_pos[i+1:]
        else: # agents_number == 1
            agents_pos_others = []
            next_agents_pos_others = []


        # エピソード中の行動履歴を保持 (他のエージェントの状態インデックス, 行動, 自エージェント位置)
        st_idx: int = self.get_index_from_states(agents_pos_others)
        # 経験データとして (他エージェント状態インデックス, 行動, 自エージェント位置, 報酬, 次の他エージェント状態インデックス, 次の自エージェント位置, doneフラグ) を保存
        next_st_idx: int = self.get_index_from_states(next_agents_pos_others)
        self.episode_data.append((st_idx, action, agent_pos, reward, next_st_idx, next_agent_pos, done))

        # エピソード終了または最大ステップに達した場合、パラメータを一括更新
        if done or step == self.max_ts:
            total_delta: float = 0.0
            num_experiences: int = len(self.episode_data)

            # 蓄積された経験データに基づいてパラメータを更新
            for exp in self.episode_data:
                st_idx_, ac, ag_pos_, rew, next_st_idx_, next_ag_pos_, is_done = exp

                # 現在の状態 (他エージェントの状態インデックスと自エージェント位置) でのQ値を取得
                phi_s_: np.ndarray = self.rbfs(ag_pos_)
                current_q: float
                if self.mask:
                    current_q = float(phi_s_.dot(self.theta_list[ac]))
                else:
                     # common_theta_list[st_idx_] はそのagent_state_idxに対応する action_size x b のtheta配列
                    if Linear.common_theta_list is None:
                         raise ValueError("Linear.common_theta_list is None")
                    current_q = float(phi_s_.dot(Linear.common_theta_list[st_idx_][ac]))


                # 次の状態での最大Q値を取得
                max_next_q_value: float = 0.0
                if not is_done:
                    # 次の自エージェント位置と次の他エージェント状態インデックスを使って次の状態でのQ値を計算
                    phi_s_next_: np.ndarray = self.rbfs(next_ag_pos_)
                    next_q_values: List[float] = []
                    for next_ac in range(self.action_size):
                        if self.mask:
                            next_q_values.append(float(phi_s_next_.dot(self.theta_list[next_ac])))
                        else:
                            if Linear.common_theta_list is None:
                                raise ValueError("Linear.common_theta_list is None")
                            next_q_values.append(float(phi_s_next_.dot(Linear.common_theta_list[next_st_idx_][next_ac])))
                    max_next_q_value = max(next_q_values)

                # TDターゲットを計算
                td_target: float = rew + (1 - is_done) * self.gamma * max_next_q_value
                # TD誤差を計算
                delta: float = td_target - current_q
                self.loss.append(delta) # TD誤差を蓄積

                # パラメータの更新 (各経験に対して行う場合)
                # self.lr * delta * phi_s_ を対応する theta に加算

            # エピソード全体の平均TD誤差に基づいて一括更新
            if num_experiences > 0:
                avg_err: float = np.mean(self.loss) #type:ignore # 蓄積されたTD誤差の平均
                 # エピソード中に記録された各ステップの経験に対してパラメータを更新
                for exp in self.episode_data:
                    st_idx_, ac, ag_pos_, rew, next_st_idx_, next_ag_pos_, is_done = exp
                    phi_s_: np.ndarray = self.rbfs(ag_pos_) # 現在の自エージェント位置に対応する基底関数値

                    if self.mask:
                        # マスクあり: 自エージェントのパラメータを更新
                        self.theta_list[ac] += self.lr * avg_err * phi_s_
                    else:
                        # マスクなし: 共有パラメータを更新
                        if Linear.common_theta_list is None:
                            raise ValueError("Linear.common_theta_list is None")
                        Linear.common_theta_list[st_idx_][ac] += self.lr * avg_err * phi_s_

                total_delta = np.sum(np.abs(self.loss)) # エピソード全体の絶対TD誤差合計


            # エピソード終了時にリストをクリア
            self.loss = []
            self.episode_data = []
            self.index_cache.clear() # キャッシュもクリア


            return total_delta / num_experiences if num_experiences > 0 else 0.0 # 平均絶対TD誤差を返す

        # エピソード中は0.0を返す (または現在のTD誤差など、設計による)
        # 線形近似では一括更新なので、ここでは0.0を返すのが適切か
        return 0.0 # エピソード中の個別の更新ではdeltaを返さない