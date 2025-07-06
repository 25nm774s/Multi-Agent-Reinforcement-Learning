"""
線形関数近似器によるQ学習の実装.
状態数だけガウス分布を生成して基底関数とする.
学習済みパラメータを用いる場合は指定CSVから読み込む.
self.sigmaはガウス基底の標準偏差(バンド幅).
"""

import csv
import numpy as np

class Linear:
    # クラス変数：他エージェントと価値共有する場合に使用
    # Q学習のパラメータ(theta)を保持
    common_theta_list = None

    def __init__(self, args, action_size):
        """
        Linearクラスのコンストラクタ.
        各種パラメータを設定し、Q値のパラメータを初期化または読み込む.

        Args:
            args: コマンドライン引数などの設定を含むオブジェクト.
            action_size: 行動空間のサイズ.
        """
        self.max_ts = args.max_timestep
        self.goals_num = args.goals_number
        self.agents_num = args.agents_number
        self.gamma = args.gamma          # 割引率
        self.lr = args.learning_rate     # 学習率
        self.grid_size = args.grid_size
        self.action_size = action_size
        self.batch_size = args.batch_size
        self.load_model = args.load_model # 0:未学習, 1:学習済みモデル, 2:真のQ値
        self.mask = args.mask             # マスク処理を行うか

        # スケール調整用ファクター (RBF計算に使用)
        self.norm_factor = np.array([1, self.grid_size]).reshape(-1, 1)

        # エピソード中のデータを一時的に保持するリスト
        self.loss = []
        self.episode_data = []
        #self.all_th_delta = [] # TODO: この変数は現在使用されていない可能性があります。確認・削除を検討.

        # 状態インデックスの計算結果をキャッシュ
        self.index_cache = {}

        # グリッド全セル数
        b = self.grid_size ** 2

        # 価値共有用のクラス変数を初期化 (Q学習のパラメータ)
        if Linear.common_theta_list is None:
            Linear.common_theta_list = [[np.zeros(b) for _ in range(action_size)]
                                        for _ in range(b**(self.agents_num-1))]

        # 基底関数用の中心 (mu) を計算 (グリッドの各セルに対応)
        self.mu_array = np.zeros((2, b))
        cnt = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.mu_array[0, cnt] = i
                self.mu_array[1, cnt] = j
                cnt += 1
        self.sigma = 0.3 # ガウス基底の標準偏差(バンド幅)

        # 行動数分のmuを保持する配列 (現状1つのみだが、拡張性を考慮しリスト形式)
        self.mu_list = [np.copy(self.mu_array)]

        # モデル読み込みまたは真の価値関数として読み込む場合
        if self.load_model == 1:
            # 学習済みモデルを読み込み
            print(f"load_model {self.load_model}: 学習済みモデルを読み込み")
            # TODO: model_pathをargsから取得するか、別の方法で指定する必要がある
            # self._load_trained_model(model_path, b)
            pass # Placeholder for loading logic
        elif self.load_model == 2:
            # 真の行動価値関数を読み込み
            print(f"load_model {self.load_model}: 真の行動価値関数を読み込み")
            # TODO: model_pathをargsから取得するか、別の方法で指定する必要がある
            # self._load_true_model(model_path, b)
            pass # Placeholder for loading logic
        else:
            # 未学習の初期化
            print(f"load_model {self.load_model}: 未学習のため初期化")
            self.theta_list = [np.zeros(b) for _ in range(self.action_size)]

    def _load_trained_model(self, model_path, b):
        """
        学習済みのQ関数パラメータをCSVから読み込んでtheta_listに格納する.

        Args:
            model_path: 学習済みモデルのCSVファイルパス.
            b: グリッドの全セル数.
        """
        with open(model_path, 'r') as f:
            rows = [np.array(row, dtype=float) for row in csv.reader(f)]
        self.theta_list = rows

        # マスクしない場合は共通パラメータとしてreshape
        if self.mask == 0:
            self.theta_list = np.array(self.theta_list)
            Linear.common_theta_list = self.theta_list.reshape(
                (b**(self.agents_num - 1), self.action_size, b)
            )

    def _load_true_model(self, model_path, b):
        """
        真の行動価値(Q)をCSVから読み込み，学習に利用する.

        Args:
            model_path: 真の行動価値のCSVファイルパス.
            b: グリッドの全セル数.
        """
        with open(model_path, 'r') as f:
            rows = [np.array(row, dtype=float) for row in csv.reader(f)]
        self.true_theta_list = rows

        # マスクしない場合，さらにreshape
        if self.mask == 0:
            self.true_theta_list = np.array(self.true_theta_list).reshape(
                (b**(self.agents_num - 1), self.action_size, b)
            )

        # Q用のtheta_listを(最低限)初期化 (真の価値をターゲットに学習するため)
        self.theta_list = [np.zeros(b) for _ in range(self.action_size)]


    def generate_states(self):
        """
        全エージェント分，セル数を考慮した状態(1次元座標のリスト)を再帰的に生成する.

        Returns:
            全エージェントの全状態の組み合わせリスト.
        """
        states = []
        def _dfs(current, depth):
            if depth == 0:
                states.append(current)
                return
            for i in range(self.grid_size):
                _dfs(current + [i], depth - 1)
        # 各エージェントの状態 (0 to grid_size-1) を組み合わせる
        _dfs([], self.agents_num)
        return states

    def generate_my_states(self):
        """
        自エージェントのみの全2次元座標を生成する.

        Returns:
            自エージェントの全可能な2次元座標リスト.
        """
        return [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)]

    def rbfs(self, state):
        """
        RBF (ガウス基底関数) を計算して返す.
        入力状態と各基底中心(mu)との距離に基づき、ガウス関数値を計算する.

        Args:
            state: 現在の自エージェントの2次元座標.

        Returns:
            各基底関数に対応する値のnumpy配列.
        """
        state = state.reshape(len(state), 1)
        # 距離(L2ノルム)を計算 (norm_factorでスケール調整)
        dist = np.linalg.norm((self.mu_list - state) / self.norm_factor, axis=1)
        # exp(-d^2/(2σ^2)) を出力
        return np.exp(-dist**2 / (2 * self.sigma**2)).ravel()

    def getQ(self, states, state, action):
        """
        自エージェントの(状態, 行動)に対応するQ値を返す.
        線形近似により Q(s,a) = φ(s)・θ_a を計算する.

        Args:
            states: 他エージェントを含む全エージェントの状態リスト.
            state: 自エージェントの2次元座標.
            action: 自エージェントの行動.

        Returns:
            計算されたQ値.
        """
        # 自エージェントの状態に対応する基底関数値を計算
        phi_s = self.rbfs(np.array(state))

        if self.mask:
            # マスクあり: 自エージェントのパラメータのみ使用
            return phi_s.dot(self.theta_list[action])
        else:
            # マスクなし: 他エージェントの状態に対応する共有パラメータを使用
            idx = self.get_index_from_states(states) # 他エージェントの状態からインデックス取得
            return phi_s.dot(Linear.common_theta_list[idx][action])

    def getTrueQ(self, states, state, action):
        """
        load_model == 2 の場合に、学習済みモデルまたは真のQ値を参照して返す.

        Args:
            states: 他エージェントを含む全エージェントの状態リスト.
            state: 自エージェントの2次元座標.
            action: 自エージェントの行動.

        Returns:
            参照された真のQ値、または利用できない場合はデフォルト値(0.0).
        """
        # 真のQ値がロードされている場合のみ有効
        if hasattr(self, 'true_theta_list'):
            phi_s = self.rbfs(np.array(state))
            if self.mask:
                return phi_s.dot(self.true_theta_list[action])
            idx = self.get_index_from_states(states)
            return phi_s.dot(self.true_theta_list[idx][action])
        else:
            # 真のQ値がロードされていない場合はエラーまたはデフォルト値を返す
            print("Warning: 真のQ値が利用できません。")
            return 0.0 # あるいは適切なデフォルト値やエラー処理を行う

    def get_index_from_states(self, states):
        """
        複数エージェントの座標(例: [x1,y1],[x2,y2]...) を用いて一意なインデックスを算出する.
        これは、他のエージェントの状態の組み合わせに対応する共有パラメータにアクセスするために使用される.
        計算結果をキャッシュして高速化.

        Args:
            states: 他エージェントの状態のリスト (自エージェントの状態は含まない).

        Returns:
            計算された一意なインデックス.
        """
        state_key = tuple(map(tuple, states))
        if state_key in self.index_cache:
            return self.index_cache[state_key]

        index = 0
        # 各エージェントの状態座標を基にインデックスを計算 (グリッドサイズを基数とする)
        for i, (x, y) in enumerate(states):
            # 各エージェントの状態がインデックスに寄与する重みを計算
            power = (self.grid_size ** (2 * (len(states) - i - 1)))
            # (x*grid_size + y) は2次元座標を1次元に変換したもの
            index += (x * self.grid_size + y) * power
        self.index_cache[state_key] = index
        return index

    def _update_with_true_q(self, agents_pos, agent_pos, action, reward, next_agents_pos, next_agent_pos, done, step):
        """
        load_model == 2 の場合のパラメータ更新処理.
        真の行動価値(Q)をターゲット (TDターゲット) として学習を行う.

        Args:
            agents_pos: 他エージェントの現在の2次元座標リスト.
            agent_pos: 自エージェントの現在の2次元座標.
            action: 自エージェントの行動.
            reward: 即時報酬.
            next_agents_pos: 他エージェントの次の2次元座標リスト.
            next_agent_pos: 自エージェントの次の2次元座標.
            done: エピソード終了フラグ.
            step: 現在のタイムステップ数.

        Returns:
            計算されたTD誤差 (delta).
        """
        # 真のQ値 (TDターゲット) を取得
        target = self.getTrueQ(agents_pos, agent_pos, action)
        # 現在のQ値を取得
        current_q = self.getQ(agents_pos, agent_pos, action)
        # TD誤差を計算
        delta = target - current_q

        # TD誤差をlossリストに蓄積 (エピソード終了時に平均を計算するため)
        self.loss.append(delta)

        # エピソード終了または最大ステップに達した場合、パラメータを一括更新
        if done or step == self.max_ts:
            avg_err = np.mean(self.loss) # 蓄積されたTD誤差の平均
            # エピソード中に記録された各ステップの経験に対してパラメータを更新
            for st_idx_, ac, ag_pos_ in self.episode_data:
                phi_s_ = self.rbfs(ag_pos_)
                if self.mask:
                    # マスクあり: 自エージェントのパラメータを更新
                    self.theta_list[ac] += self.lr * avg_err * phi_s_
                else:
                    # マスクなし: 共有パラメータを更新
                    Linear.common_theta_list[st_idx_][ac] += self.lr * avg_err * phi_s_

        return delta # 計算されたTD誤差を返す

    def update(self, i, states, action, reward, next_state, done, step):
        """
        TD誤差を計算し, エピソード終了時に一括でパラメータを更新するメインメソッド.
        load_model の設定に応じて異なる更新ロジックを呼び出す.

        Args:
            i: 自エージェントのインデックス.
            states: (goals + agents) のタプル形式の現在の状態.
            action: 自エージェントの行動.
            reward: 即時報酬.
            next_state: (goals + agents) のタプル形式の次の状態.
            done: エピソード終了フラグ.
            step: 現在のタイムステップ数.

        Returns:
            計算されたTD誤差 (delta).
        """
        # 状態からゴール位置とエージェント位置を分離
        goals_pos = list(states[:self.goals_num])
        agents_pos = list(states[self.goals_num:])
        agent_pos = np.array(agents_pos[i]) # 自エージェントの現在の位置

        # 次の状態からゴール位置とエージェント位置を分離
        next_goals_pos = list(next_state[:self.goals_num])
        next_agents_pos = list(next_state[self.goals_num:])
        next_agent_pos = np.array(next_agents_pos[i]) # 自エージェントの次の位置

        # 他エージェントの状態リストを作成 (自エージェントの状態を除去)
        agents_pos_others = agents_pos.copy()
        agents_pos_others.pop(i)
        next_agents_pos_others = next_agents_pos.copy()
        next_agents_pos_others.pop(i)

        # エピソード中の行動履歴を保持 (他のエージェントの状態インデックス, 行動, 自エージェントの位置)
        st_idx = self.get_index_from_states(agents_pos_others)
        self.episode_data.append((st_idx, action, agent_pos))

        # エピソードの最初のステップでリストを初期化
        if step == 0:
            self.loss = []
            self.episode_data = []
            #self.all_th_delta = [] # TODO: この変数は現在使用されていない可能性があります。確認・削除を検討.


        # load_model の設定に基づいて更新ロジックを選択
        if self.load_model in [0, 1]:
            # 未学習 (0) または学習済みモデルを使用 (1) の場合
            # 標準的なQ学習の更新 (TDターゲットに max Q(s',a') を使用)

            # 次の状態での最大Q値を取得
            next_q = [self.getQ(next_agents_pos_others, next_agent_pos, a) for a in range(self.action_size)]
            # TDターゲットを計算: 報酬 + 割引率 * max Q(s',a')
            target = reward + (1 - done) * self.gamma * max(next_q)
            # 現在のQ値を取得
            current_q = self.getQ(agents_pos_others, agent_pos, action)
            # TD誤差を計算
            delta = target - current_q
            # TD誤差をlossリストに蓄積
            self.loss.append(delta)

            # エピソード終了または最大ステップに達した場合、パラメータを一括更新
            if done or step == self.max_ts:
                avg_err = np.mean(self.loss) # 蓄積されたTD誤差の平均
                # エピソード中に記録された各ステップの経験に対してパラメータを更新
                for st_idx_, ac, ag_pos_ in self.episode_data:
                    phi_s_ = self.rbfs(ag_pos_)
                    if self.mask:
                        # マスクあり: 自エージェントのパラメータを更新
                        self.theta_list[ac] += self.lr * avg_err * phi_s_
                    else:
                        # マスクなし: 共有パラメータを更新
                        Linear.common_theta_list[st_idx_][ac] += self.lr * avg_err * phi_s_

        elif self.load_model == 2:
            # load_model == 2 の場合: 真の価値関数をターゲットとして学習
            # 新しいプライベートメソッドを呼び出す
            delta = self._update_with_true_q(agents_pos_others, agent_pos, action, reward, next_agents_pos_others, next_agent_pos, done, step)

        else:
            # 未定義の load_model 値の場合
            print(f"Warning: 未定義の load_model 値 ({self.load_model}) です。更新処理はスキップされます。")
            delta = 0.0 # デフォルトのdelta値

        # キャッシュクリア（必要に応じて、例: 一定ステップごと）
        # if step != 0 and step % 10000 == 0:
        #     self.index_cache.clear()

        return delta