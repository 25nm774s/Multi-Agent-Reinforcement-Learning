"""
線形関数近似器によるQ学習の実装.
状態数だけガウス分布を生成して基底関数とする.
学習済みパラメータを用いる場合は指定CSVから読み込む.
self.sigmaはガウス基底の標準偏差(バンド幅).
"""

import os
import csv
import numpy as np
import itertools

class Linear:
    # クラス変数：他エージェントと価値共有する場合に使用
    common_theta_list = None
    common_state_theta_list = None

    def __init__(self, args, action_size, model_path):
        self.max_ts = args.max_timestep
        self.goals_num = args.goals_number
        self.agents_num = args.agents_number
        self.gamma = args.gamma
        self.lr = args.learning_rate
        self.cell_num = args.cell_number
        self.action_size = action_size
        self.batch_size = args.batch_size
        self.load_model = args.load_model
        self.mask = args.mask
        self.learning_mode = args.learning_mode

        # スケール調整用
        self.norm_factor = np.array([1, self.cell_num]).reshape(-1, 1)

        self.loss = []
        self.episode_data = []
        self.all_th_delta = []
        self.index_cache = {}

        # グリッド全セル数
        b = self.cell_num ** 2

        # 価値共有用のクラス変数を初期化
        if Linear.common_theta_list is None:
            Linear.common_theta_list = [[np.zeros(b) for _ in range(action_size)]
                                        for _ in range(b**(self.agents_num-1))]
        if Linear.common_state_theta_list is None:
            Linear.common_state_theta_list = [np.zeros(b) for _ in range(b**(self.agents_num-1))]

        # 基底関数用の中心 (mu) を計算
        self.mu_array = np.zeros((2, b))
        cnt = 0
        for i in range(self.cell_num):
            for j in range(self.cell_num):
                self.mu_array[0, cnt] = i
                self.mu_array[1, cnt] = j
                cnt += 1
        self.sigma = 0.3 # バンド幅元々0.5

        # 行動数分のmuを保持する配列(現状1つのみ)
        self.mu_list = [np.copy(self.mu_array)]

        # モデル読み込みまたは真の価値関数として読み込む場合
        if self.load_model == 1:
            # 学習済みモデルを読み込み
            self._load_trained_model(model_path, b)
        elif self.load_model == 2:
            # 真の価値関数を用いる場合
            self._load_true_model(model_path, b)
        else:
            # 未学習の初期化
            self.theta_list = [np.zeros(b) for _ in range(self.action_size)]

    def _load_trained_model(self, model_path, b):
        """学習済みのQ関数パラメータをCSVから読み込んでtheta_listに格納"""
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
        """真の行動価値(Q)または状態価値(V)を読み込み，学習に利用する"""
        # 学習モードがVの場合はファイル名上のQをVに置換して読む
        if self.learning_mode == 'V':
            v_path = model_path.replace('V', 'Q', 1)
        else:
            v_path = model_path

        # 読み込んだ行動価値を true_theta_list に保管
        with open(v_path, 'r') as f:
            rows = [np.array(row, dtype=float) for row in csv.reader(f)]
        self.true_theta_list = rows

        # マスクしない場合，さらにreshape
        if self.mask == 0:
            self.true_theta_list = np.array(self.true_theta_list).reshape(
                (b**(self.agents_num - 1), self.action_size, b)
            )

            # 状態価値関数Vとして学習する場合 (行動価値→状態価値へ変換)
            if self.learning_mode == 'V':
                self._prepare_true_state_value(model_path, b)

        # Q用のtheta_listを(最低限)初期化
        self.theta_list = [np.zeros(b) for _ in range(self.action_size)]

    def _prepare_true_state_value(self, model_path, b):
        """真の行動価値を基に状態価値を計算し，ファイルに保存する"""
        # Q→Vに置き換えたファイルパス
        v_path = model_path.replace('Q', 'V', 1)

        # 既に学習済みのVがあればロード，無ければ作成
        if os.path.exists(v_path):
            with open(v_path, 'r') as f:
                self.true_common_state_theta_list = [np.array(row, dtype=float)
                                                     for row in csv.reader(f)]
        else:
            print("行動価値を状態価値に変換中...")
            self.true_common_state_theta_list = [np.zeros(b)
                                                 for _ in range(b**(self.agents_num - 1))]

            states = self.generate_states()        # 全エージェントの全状態
            my_state = self.generate_my_states()   # 自エージェントのみの全状態
            state_combinations = list(itertools.product(states, repeat=self.agents_num))
            state_combinations = [list(map(list, comb)) for comb in state_combinations]

            # 行動価値から状態価値を算出して保存（大きな計算になる可能性あり）
            #for idx in range(b**self.agents_num):
            #    for j in range(self.agents_num):
            #        tmp_states = state_combinations[idx].copy()
            #        tmp_states.pop(j)

            # 軽くなりそうなやつ
            for idx, comb in enumerate(itertools.product(states, repeat=self.agents_num)):
                temp_states = list(map(list, comb))
                for j in range(self.agents_num):
                    tmp_states = temp_states.copy()
                    tmp_states.pop(j)

                    for l in range(b):
                        val = self.compute_state_value_from_Q(tmp_states, my_state[l])
                        self.true_common_state_theta_list[idx][l] = val
                print(idx)  # 計算の進捗表示

            os.makedirs(os.path.dirname(v_path), exist_ok=True)
            with open(v_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in self.true_common_state_theta_list:
                    writer.writerow(row)

    def compute_state_value_from_Q(self, states, agent_pos, epsilon=0.1):
        """行動価値 Q(s,a) から e-greedy 方策に基づく状態価値 V(s) を計算"""
        qs = [self.getTrueQ(states, agent_pos, a) for a in range(self.action_size)]
        max_q = max(qs)
        max_a = np.argmax(qs)

        # (1 - epsilon + epsilon/|A|) * maxQ
        first_term = (1.0 - epsilon + epsilon/self.action_size) * max_q

        # 残りの行動 (a != max_a) についての期待値
        second_term = sum(q for i, q in enumerate(qs) if i != max_a)
        second_term *= (epsilon / self.action_size)

        return first_term + second_term

    def generate_states(self):
        """全エージェント分，セル数を考慮した状態(1次元)を再帰的に生成"""
        states = []
        def _dfs(current, depth):
            if depth == 0:
                states.append(current)
                return
            for i in range(self.cell_num):
                _dfs(current + [i], depth - 1)
        _dfs([], self.agents_num)
        return states

    def generate_my_states(self):
        """自エージェントのみの全2次元座標を生成"""
        return [[i, j] for i in range(self.cell_num) for j in range(self.cell_num)]

    def rbfs(self, state):
        """RBF (ガウス基底関数) を計算して返す"""
        state = state.reshape(len(state), 1)
        # 距離(L2ノルム)を計算
        dist = np.linalg.norm((self.mu_list - state) / self.norm_factor, axis=1)
        # exp(-d^2/(2σ^2)) を出力
        return np.exp(-dist**2 / (2 * self.sigma**2)).ravel()

    def getV(self, states, state):
        idx = self.get_index_from_states(states)
        return self.rbfs(np.array(state)).dot(Linear.common_state_theta_list[idx])

    def getTrueV(self, states, state):
        idx = self.get_index_from_states(states)
        return self.rbfs(np.array(state)).dot(self.true_common_state_theta_list[idx])

    def getQ(self, states, state, action):
        """自エージェントの(状態, 行動)に対応するQ値を返す"""
        if self.mask:
            return self.rbfs(np.array(state)).dot(self.theta_list[action])
        idx = self.get_index_from_states(states)
        return self.rbfs(np.array(state)).dot(Linear.common_theta_list[idx][action])

    def getTrueQ(self, states, state, action):
        """学習済みモデルを真のQ値として参照する場合"""
        if self.mask:
            return self.rbfs(np.array(state)).dot(self.true_theta_list[action])
        idx = self.get_index_from_states(states)
        return self.rbfs(np.array(state)).dot(self.true_theta_list[idx][action])

    def get_index_from_states(self, states):
        """
        複数エージェントの座標(例: [x1,y1],[x2,y2]...) を用いて一意なインデックスを算出.
        キャッシュして高速化.
        """
        state_key = tuple(map(tuple, states))
        if state_key in self.index_cache:
            return self.index_cache[state_key]

        index = 0
        for i, (x, y) in enumerate(states):
            power = (self.cell_num ** (2 * (len(states) - i - 1)))
            index += (x * self.cell_num + y) * power
        self.index_cache[state_key] = index
        return index

    def update(self, i, states, action, reward, next_state, done, step):
        """
        TD誤差を計算し, エピソード終了時に一括でパラメータを更新する.
        i: 自エージェントのインデックス
        states: (goals + agents) のタプル
        action: 自エージェントの行動
        reward: 即時報酬
        next_state: 次の状態(タプル)
        done: エピソード終了フラグ
        step: 現在のタイムステップ数
        """
        goals_pos = list(states[:self.goals_num])
        agents_pos = list(states[self.goals_num:])
        agent_pos = np.array(agents_pos[i])

        next_goals_pos = list(next_state[:self.goals_num])
        next_agents_pos = list(next_state[self.goals_num:])
        next_agent_pos = np.array(next_agents_pos[i])

        # 他エージェントの状態を除去
        agents_pos.pop(i)
        next_agents_pos.pop(i)

        # エピソード中の行動履歴を保持
        st_idx = self.get_index_from_states(agents_pos)
        self.episode_data.append((st_idx, action, agent_pos))

        if step == 0:
            self.loss = []
            self.episode_data = []
            self.all_th_delta = []

        # 未学習 or 学習済みモデルをそのまま使う場合
        if self.load_model in [0, 1]:
            next_q = [self.getQ(next_agents_pos, next_agent_pos, a) for a in range(self.action_size)]
            target = reward + (1 - done) * self.gamma * max(next_q)
            current_q = self.getQ(agents_pos, agent_pos, action)
            delta = target - current_q
            self.loss.append(delta)

            # エピソード終了または最大ステップで一括更新
            if done or step == self.max_ts:
                avg_err = np.mean(self.loss)
                for st_idx_, ac, ag_pos_ in self.episode_data:
                    if self.mask:
                        self.theta_list[ac] += self.lr * avg_err * self.rbfs(ag_pos_)
                    else:
                        Linear.common_theta_list[st_idx_][ac] += self.lr * avg_err * self.rbfs(ag_pos_)

        else:
            # load_model == 2 : 真の価値関数として改めて学習する
            if self.learning_mode == 'V':
                # V学習時は (trueV - currentV)^2 の誤差を蓄積して最後にまとめて反映
                states_all = self.generate_states()
                my_state_all = self.generate_my_states()
                combos = list(itertools.product(states_all, repeat=self.agents_num))
                combos = [list(map(list, c)) for c in combos]

                for idx_c in range(self.agents_num):
                    for j_c in range(self.agents_num):
                        tmp = combos[idx_c].copy()
                        tmp.pop(j_c)
                        for l in range(self.cell_num**2):
                            tv = self.getTrueV(tmp, my_state_all[l])
                            cv = self.getV(tmp, my_state_all[l])
                            self.loss.append(tv - cv)               # 学習用
                            self.all_th_delta.append((tv - cv)**2)  # 数値実験でloss出す用

                delta = self.all_th_delta

            else:
                # Q学習の場合 (trueQ - currentQ)
                target = self.getTrueQ(agents_pos, agent_pos, action)
                current_q = self.getQ(agents_pos, agent_pos, action)
                delta = target - current_q

            if done or step == self.max_ts:
                avg_err = np.mean(self.loss)

                # 状態価値更新
                if self.learning_mode == 'V':
                    for st_idx_, ac, ag_pos_ in self.episode_data:
                        Linear.common_state_theta_list[st_idx_] += self.lr * avg_err * self.rbfs(ag_pos_)

                # 行動価値更新
                if self.learning_mode == 'Q':
                    for st_idx_, ac, ag_pos_ in self.episode_data:
                        if self.mask:
                            self.theta_list[ac] += self.lr * avg_err * self.rbfs(ag_pos_)
                        else:
                            Linear.common_theta_list[st_idx_][ac] += self.lr * avg_err * self.rbfs(ag_pos_)

        # キャッシュクリア（一応）
        #if step != 0 and step % 10000 == 0:
        #    self.index_cache.clear()

        return delta