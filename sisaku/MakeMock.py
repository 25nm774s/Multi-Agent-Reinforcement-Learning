import sys
import torch
import os
import csv
import numpy as np
from collections import deque
import random # ReplayBuffer mock needs random

# from env import GridWorld # 環境クラスは別途定義されていると仮定
# from DQN.Agent_DQN import Agent_DQN # エージェントクラスは別途定義されていると仮定
# from utils.plot_results import PlotResults # プロット関連のクラスは別途定義されていると仮定
# from utils.Model_Saver import Saver # モデル保存関連のクラスは別途定義されていると仮定

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# 以下、必要なモックオブジェクトを定義します。
# 実際のコードでは、これらのクラスやオブジェクトは適切にインポートまたは初期化される必要があります。

class MockGridWorld:
    """
    GridWorld クラスのモック.
    実際の環境クラスの代わりに、簡易的な動作を提供します。
    """
    def __init__(self, args):
        """
        MockGridWorld コンストラクタ.

        Args:
            args: 環境設定を含むオブジェクト.
                  (grid_size, agents_number, goals_num 属性を持つことを想定)
        """
        self.grid_size = args.grid_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_num
        # モックとして初期状態の例を持つ
        self._mock_goals = [(0, 0), (self.grid_size-1, self.grid_size-1)][:self.goals_num]
        self._mock_agents = [(1, 1), (2, 2)][:self.agents_num] # サンプルの初期位置

    def reset(self):
        """
        環境をリセットし、全体状態（ゴール＋エージェント位置）を返すモック.
        エージェントの位置は毎回サンプルとする.
        """
        # ゴール位置は固定、エージェント位置は毎回サンプルとする
        mock_global_state = tuple(self._mock_goals + self._mock_agents)
        print("MockGridWorld: reset called, returning mock global_state.")
        return mock_global_state

    def step(self, global_state, actions):
        """
        行動に基づいて状態を更新し、報酬と完了フラグを返すモック.
        状態は変化しない、報酬はゼロ、完了しないという単純なモック動作をします。

        Args:
            global_state (Any): 現在の全体状態.
            actions (list): 各エージェントの行動リスト.

        Returns:
            tuple: (next_global_state, reward, done) のタプル.
                   next_global_state (Any): 次の全体状態.
                   reward (float): 得られた報酬.
                   done (bool): エピソード完了フラグ.
        """
        # 簡単な状態遷移のモック
        next_global_state = global_state # 状態は変化しないとする単純なモック
        reward = 0 # 報酬はゼロとするモック
        done = False # 完了しないとするモック
        print(f"MockGridWorld: step called with global_state={global_state}, actions={actions}. Returning mock next_global_state, reward, done.")
        return next_global_state, reward, done

class MockAgent_DQN:
    """Agent_DQN クラスのモック"""
    def __init__(self, args):
        self.action_size = 5 # 行動空間のサイズをモックとして定義
        self.epsilon = 1.0 # ε-greedyのためのεをモックとして定義
        self.model = MockDQNModel(args) # 内部にモックモデルを持つ
        self.replay_buffer = MockReplayBuffer(args) # 内部にモックリプレイバッファを持つ
        self.device = torch.device(args.device)

    def get_action(self, i, global_state):
        """行動を選択するモック"""
        # ランダムに行動を選択するモック
        action = np.random.choice(self.action_size)
        #print(f"MockAgent_DQN[{i}]: get_action called. Returning mock action {action}.")
        return action

    def decay_epsilon_power(self, step: int, alpha=0.5):
        """εを減衰させるモック"""
        #print(f"MockAgent_DQN: decay_epsilon_power called with step {step}.")
        pass # 何もしないモック

    def observe_and_store_experience(self, global_state, action, reward, next_global_state, done):
        """経験を保存するモック"""
        self.replay_buffer.add(global_state, action, reward, next_global_state, done)
        #print("MockAgent_DQN: observe_and_store_experience called.")
        pass # 何もしないモック

    def learn_from_experience(self, i, episode_num):
        """経験から学習するモック"""
        # 学習ロジックの代わりにNoneを返すモック (学習しない)
        #print(f"MockAgent_DQN[{i}]: learn_from_experience called.")
        # バッチサイズを満たしていればNone以外を返すようにしても良い
        # if len(self.replay_buffer) >= self.replay_buffer.batch_size:
        #    return 0.0 # モックの損失値
        return None # 学習しないモック

    def loading_model(self, model_path):
        """モデルをロードするモック"""
        print(f"MockAgent_DQN: loading_model called with path {model_path}.")
        pass # 何もしないモック


class MockPlotResults:
    """
    PlotResults クラスのモック.
    プロット関連の動作の代わりに、メッセージを出力します。
    """
    def __init__(self, save_dir):
        """
        MockPlotResults コンストラクタ.

        Args:
            save_dir (str): 結果を保存するディレクトリパス.
        """
        self.save_dir = save_dir
        print(f"MockPlotResults initialized with save_dir: {save_dir}")

    def draw(self):
        """結果をプロットするモック."""
        print("MockPlotResults: draw called.")
        pass # 何もしないモック

    def draw_heatmap(self, grid_size):
        """ヒートマップをプロットするモック."""
        print(f"MockPlotResults: draw_heatmap called with grid_size: {grid_size}.")
        pass # 何もしないモック

class MockSaver:
    """
    Saver クラスのモック.
    ログ保存やモデル保存の動作の代わりに、メッセージを出力します。
    """
    def __init__(self, save_dir):
        """
        MockSaver コンストラクタ.

        Args:
            save_dir (str): 保存ディレクトリパス.
        """
        self.save_dir = save_dir
        print(f"MockSaver initialized with save_dir: {save_dir}")

    def log_agent_states(self, episode, time_step, agent_id, agent_state):
        """エージェントの状態をログに記録するモック."""
        #print(f"MockSaver: log_agent_states called for episode {episode}, step {time_step}, agent {agent_id}.")
        pass # 何もしないモック

    def log_scores(self, episode, time_step, reward, loss):
        """スコアをログに記録するモック."""
        #print(f"MockSaver: log_scores called for episode {episode}, step {time_step}.")
        pass # 何もしないモック

    def save_dqn_weights(self, agents):
        """DQNの重みを保存するモック."""
        print("MockSaver: save_dqn_weights called.")
        pass # 何もしないモック

    def save_q_table(self, agents, mask):
        """Qテーブルを保存するモック."""
        print("MockSaver: save_q_table called.")
        pass # 何もしないモック

# 以下は ReplayBuffer, DQNModel, QNet のモック。既に存在するクラスと名前が重複しないように Mock_ を付けます。
class MockReplayBuffer:
    """ReplayBuffer クラスのモック"""
    def __init__(self, args):
        self.buffer = deque(maxlen=args.buffer_size)
        self.batch_size = args.batch_size
        self.device = torch.device(args.device)
        print("MockReplayBuffer initialized.")

    def add(self, global_state, action, reward, next_global_state, done):
        """経験をバッファに追加するモック"""
        data = (global_state, action, reward, next_global_state, done)
        self.buffer.append(data)
        #print("MockReplayBuffer: add called.")

    def __len__(self):
        """バッファサイズを返すモック"""
        return len(self.buffer)

    def get_batch(self):
        """バッチを取得するモック"""
        if len(self.buffer) < self.batch_size:
             return None # バッチサイズに満たない場合は None を返す
        data = random.sample(self.buffer, self.batch_size)
        # モックなのでテンソル変換は省略、そのまま返す
        global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch = zip(*data)
        #print("MockReplayBuffer: get_batch called.")
        return global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch # タプルのまま返す

class MockQNet(torch.nn.Module):
    """QNet クラスのモック"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        print(f"MockQNet initialized with input_size={input_size}, output_size={output_size}.")

    def forward(self, x):
        """順伝播のモック"""
        # 入力と同じバッチサイズで、出力サイズのランダムなテンソルを返すモック
        # 入力 x の形状が (batch_size, input_size) であると仮定
        if isinstance(x, list) or isinstance(x, tuple): # get_actionなどで単一の状態でリストやタプルで渡ってくる場合に対応
             batch_size = len(x)
             # モックなので、内部要素がテンソルかどうかの判定はせず、要素数でバッチサイズとみなす
        elif isinstance(x, torch.Tensor):
             batch_size = x.size(0)
        else:
             batch_size = 1 # その他の単一入力の場合

        # デバイスを合わせる (入力がテンソルでない場合はCPUを仮定)
        if isinstance(x, torch.Tensor):
             device = x.device
        else:
             device = torch.device("cpu") # デフォルトはCPU

        mock_output = torch.randn(batch_size, self.output_size, device=device)
        #print(f"MockQNet: forward called with input shape {x.shape if isinstance(x, torch.Tensor) else type(x)}. Returning mock output shape {mock_output.shape}.")
        return mock_output

class MockDQNModel:
    """DQNModel クラスのモック"""
    def __init__(self, args):
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.agents_num = args.agents_number
        self.goals_num = args.goals_num
        self.load_model = args.load_model
        self.mask = args.mask
        self.lr = args.learning_rate
        self.action_size = 5
        self.target_update_frequency = getattr(args, 'target_update_frequency', 100)

        # mask設定に応じた入力サイズ計算のモック
        # マスクモード時は自身の位置(x,y)で2次元、非マスク時は全体状態の次元 ((goals+agents)*2)
        input_size = 2 if self.mask else (self.agents_num + self.goals_num) * 2
        self.qnet = MockQNet(input_size, self.action_size)
        self.qnet_target = MockQNet(input_size, self.action_size) # ターゲットネットワークもモック

        # オプティマイザはモックでは不要だが、形だけ定義しても良い
        # self.optimizer = None # モックなのでオプティマイザは使わない

        print("MockDQNModel initialized.")


    def update(self, i: int, global_states_batch, actions_batch, rewards_batch, next_global_states_batch, dones_batch, episode_num: int):
        """Qネットワークの更新ロジックのモック"""
        # 学習済みモデル使用時は学習しないモック
        if self.load_model == 1:
            return None

        # バッチデータがNoneの場合は学習しないモック
        if global_states_batch is None:
             return None

        # ここで global_states_batch からエージェント i の状態を抽出 (モックなので実際には使わないが、概念として)
        # agent_states_batch = self._extract_agent_state(i, global_states_batch)
        # next_agent_states_batch = self._extract_agent_state(i, next_global_states_batch)

        print(f"MockDQNModel: update called for agent {i}, episode {episode_num}.")
        # 損失値のモックを返す (例: ランダムな値や固定値)
        mock_loss = np.random.rand() * 0.1 # 0.0から0.1の間のランダムな損失
        #print(f"MockDQNModel: Returning mock loss {mock_loss}.")
        return mock_loss

    def sync_qnet(self) -> None:
        """ターゲットネットワーク同期のモック"""
        print("MockDQNModel: sync_qnet called.")
        pass # 何もしないモック

    # モックとして抽出メソッドを定義 (updateからは呼ばないが、概念を示すため)
    def _extract_agent_state(self, i: int, global_states_batch):
         """
         全体状態バッチから、特定エージェントの状態バッチを抽出するモック
         """
         if not self.mask:
              return global_states_batch # マスクしない場合は全体状態

         # モックの全体状態バッチはタプルのタプルを想定: ((g1),(g2),...,(a1),(a2),...)
         # 抽出したいのは各バッチサンプルの、特定エージェントの状態に対応する部分
         # global_states_batch は zip(*data) で得られたタプルなので、形状は (batch_size,) で各要素が全体状態タプル
         # 各全体状態タプルからエージェント i の位置を抽出
         agent_states = [global_state[self.goals_num + i] for global_state in global_states_batch]
         return agent_states # モックなのでNumPy/Tensor変換はしない

# MultiAgent_DQN クラスの実装（修正）
class MultiAgent_DQN:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, agents: list[MockAgent_DQN]): # 型ヒントを MockAgent_DQN に変更
        """
        MultiAgent_DQN クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_num, grid_size, load_model, mask,
                   save_agent_states 属性を持つことを想定)
            agents (list[MockAgent_DQN]): 使用するエージェントオブジェクトのリスト.
        """
        # 実際の GridWorld ではなくモックを使用
        self.env = MockGridWorld(args)
        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_num
        self.grid_size = args.grid_size
        self.load_model = args.load_model
        self.mask = args.mask

        self.save_agent_states = args.save_agent_states

        save_dir = os.path.join(
            "output",
            f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]"
        )
        # ディレクトリがなければ作成
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 実際の Saver, PlotResults ではなくモックを使用
        self.saver = MockSaver(save_dir)
        self.plot_results = MockPlotResults(save_dir)


    def run(self):
        """
        強化学習のメイン実行ループ.
        指定されたエピソード数だけ環境とのインタラクションと学習を行います。
        """
        # 事前条件チェック
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}DQN{RESET} で学習中...\n")

        total_step = 0
        avg_reward_temp, avg_step_temp = 0, 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            print('■', end='',flush=True)  # 進捗表示

            # 100エピソードごとに平均を出力
            if episode_num % 100 == 0:
                print()
                avg_reward = avg_reward_temp / 100
                avg_step = avg_step_temp / 100
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward}{RESET}\n")
                avg_reward_temp, avg_step_temp = 0, 0

            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # ここで返される global_state は全体状態
            current_global_state = self.env.reset()

            done = False
            step_count = 0
            ep_reward = 0

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions = []
                for i, agent in enumerate(self.agents):
                    # エージェントにε減衰を適用
                    agent.decay_epsilon_power(total_step)

                    # エージェントに行動を選択させる際に、現在の全体状態(current_global_state)を渡す
                    # エージェント内部で自身の観測(masking)を行う
                    actions.append(agent.get_action(i, current_global_state))

                # エージェントの状態を保存（オプション）
                # current_global_state のエージェント部分を切り出して保存
                if self.save_agent_states:
                     # current_global_state の構造に依存してエージェント部分を抽出
                     # モックの構造に合わせて、agent_pos は current_global_state の goals_num 以降とする
                    agent_positions_in_global_state = current_global_state[self.goals_num:]
                    for i, agent_pos in enumerate(agent_positions_in_global_state):
                        self.saver.log_agent_states(episode_num, step_count, i, agent_pos)

                # 環境にステップを与えて状態を更新
                # 入力に現在の全体状態 current_global_state を使用
                next_global_state, reward, done = self.env.step(current_global_state, actions)

                # DQNは逐次更新 (経験をバッファに保存し、後でバッチで学習)
                losses = []
                for i, agent in enumerate(self.agents):
                    # エージェントは自身の経験(状態s, 行動a, 報酬r, 次状態s', 終了フラグdone)をストア
                    # ここでも状態sと次状態s'は環境全体の全体状態を渡す
                    agent.observe_and_store_experience(current_global_state, actions[i], reward, next_global_state, done)

                    # 学習は別のタイミングでトリガー (通常はバッファが一杯になったら、または定期的に)
                    # learn_from_experience はバッファからバッチを取得し、モデルの update を呼び出す
                    # モデルの update にはバッチデータが渡される
                    loss = agent.learn_from_experience(i, episode_num)
                    if loss is not None:
                        losses.append(loss)

                current_global_state = next_global_state # 全体状態を更新

                # lossesがNoneでないものだけを抽出して平均を計算 (この場所での損失計算は逐次更新には合わない可能性)
                # 通常は learn_from_experience の呼び出し元 (ここ) で損失を収集・平均化するのではなく、
                # learn_from_experience 内部で計算された損失を返すようにし、ここで収集・平均化する。
                # 現在の learn_from_experience モックは None を返すので、lossesは空になる。
                # 実装に合わせて修正が必要。
                # avg_loss の計算をエピソードの最後に移動するか、learn_from_experience が学習時のみ損失を返すように調整する。
                # ここではエピソードの最後に移動する前提で、lossesリストのクリアはしない。

                step_count += 1
                total_step += 1

            # エピソード終了
            # lossesがNoneでないものだけを抽出して平均を計算 (エピソード全体での平均損失を計算する場合)
            # 逐次更新の場合、エピソードの最後に損失を平均するのは一般的ではない。
            # 各学習ステップで計算された損失をログに残す方が多い。
            # ここでは仮に losses リストに集められたものがあれば平均を出す。
            valid_losses = [l for l in losses if l is not None]
            avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else 0 # lossesが空の場合は0

            # ログにスコアを記録 (エピソードごとのステップ数、報酬、およびそのエピソード中の平均損失)
            self.saver.log_scores(episode_num, step_count, ep_reward, avg_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行

    def save_model_weights(self):
        """学習済みモデルの重みを保存する."""
        # モデル保存やプロット
        self.saver.save_dqn_weights(self.agents)

    def save_Qtable(self):
        """
        Qテーブルを保存する (NNベースの場合は近似計算が必要な場合がある).
        現在の実装はNNベースのため、このメソッドは必要ないか、
        Q関数からQ値を計算して保存するなどの実装が必要。
        """
        # エージェントがNNベースなのでQテーブルは保存しない (または、Q関数から近似的に計算して保存するなど)
        # 現在の実装では NN ベースなので save_Qtable は不要かもしれない。
        # モックとしては存在させる。
        print("MultiAgent_DQN: save_Qtable called (for NN based agents).")
        # self.saver.save_q_table(self.agents,self.mask) # Saverのモックを呼ぶ
        pass

    def result_show(self):
        """学習結果をプロットして表示する."""
        self.plot_results.draw()
        self.plot_results.draw_heatmap(self.grid_size)


# モックに必要な args オブジェクトを定義
# 実際の実行時には、適切な引数を持つ args オブジェクトが渡される必要があります。
class MockArgs:
    def __init__(self):
        self.grid_size = 5
        self.agents_number = 2
        self.goals_number = 1
        self.reward_mode = 2
        self.render_mode = False
        self.episode_number = 10
        self.max_timestep = 100
        self.load_model = 0
        self.mask = True
        self.save_agent_states = True
        self.buffer_size = 10000
        self.batch_size = 32
        self.optimizer = 'Adam'
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.decay_epsilon = 10000 # ε減衰ステップ数のモック
        self.device = 'cpu' # または 'cuda' if torch.cuda.is_available() else 'cpu'
        self.target_update_frequency = 100

# 使用例のモック
# if __name__ == '__main__':
#     args = MockArgs()
#     # モックエージェントのリストを作成
#     mock_agents = [MockAgent_DQN(args) for _ in range(args.agents_number)]
#     # MultiAgent_DQN をモックと共に初期化
#     multi_agent_dqn = MultiAgent_DQN(args, mock_agents)
#     # run メソッドを実行 (モックの動作を確認)
#     multi_agent_dqn.run()
#     # 結果表示やモデル保存のモックを呼び出す
#     multi_agent_dqn.save_model_weights()
#     multi_agent_dqn.result_show()