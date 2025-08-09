# src/Q/MultiAgent_Q.py

# MultiAgentBase 基底クラスをインポート
# プロジェクト構造に合わせてパスを調整してください
from multi_agent_base import MultiAgentBase

# 必要なライブラリをインポート
import os
import sys

# Agent クラスのインポート (src/Q パッケージ内)
from .Agent_Q import Agent # 相対インポートの場合

# 共通ユーティリティクラスのインポート（src/utils パッケージ）
# プロジェクト構造に合わせてパスを調整してください
from utils.saver import Saver
from utils.plot_results import PlotResults

# 環境クラスのインポート（src/envs パッケージなど）
# プロジェクト構造に合わせてパスを調整してください
from envs.grid_world import GridWorld

# カラーコード（必要であれば）
GREEN = '\033[92m'
RESET = '\033[0m'


# MultiAgentBase を継承
class MultiAgent_Q(MultiAgentBase):
    # agents の型ヒントをより具体的にすることも可能 (Agent クラスが AgentBase を継承している場合など)
    def __init__(self, args, agents:list[Agent]):
        """
        MultiAgent_Q クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
            agents (list[Agent]): 使用する Agent_Q オブジェクトのリスト.
        """
        # 基底クラスの初期化を呼び出す
        super().__init__(args, agents)

        # Q学習固有の初期化や、基底クラスで定義しなかった共通処理
        # 結果保存ディレクトリの設定と作成 (アルゴリズム固有情報を含む)
        save_dir = os.path.join(
            "output",
            f"Q_mask[{self.mask}]_Reward[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_num}]"
        )
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # SaverとPlotResultsのインスタンス化は派生クラスで行う
        self.saver = Saver(save_dir=save_dir)
        self.plot_results = PlotResults(save_dir)

        # argsにdir_path属性を追加 (Agentクラスの初期化で使用される)
        # これは Agent_Q の初期化前に必要かもしれないので、main.py で設定するか、
        # Agent の初期化を MultiAgent_Q の中で行う設計にするか検討
        # 現在のmain.pyの構造だと、agentsリストを作成してからMultiAgent_Qを初期化しているので、
        # args.dir_pathはmain.pyで設定しておく方が自然です。
        # args.dir_path = save_dir # これはmain.pyで設定するのが自然

        # 事前条件チェックは基底クラスで行われている


    # MultiAgentBase の run 抽象メソッドを実装
    def run(self):
        """
        Q学習のメイン実行ループ.
        """
        # 事前条件チェックは基底クラスで行われている

        # 学習開始メッセージ
        if self.mask:
            print(f"{GREEN}IQL/CQL (Q学習ベース) で学習中{RESET}\n")
        else:
            print(f"{GREEN}Q学習で学習中{RESET}\n")

        total_step = 0
        avg_reward_temp, avg_step_temp = 0, 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            # print('■', end='',flush=True)  # 進捗表示 - コメントアウトしてログを見やすくする
            if episode_num % 10 == 0:
                print(f"Episode {episode_num} / {self.episode_num}", end='\r', flush=True)

            # 100エピソードごとに平均を出力
            if episode_num % 100 == 0:
                print() # 改行して進捗表示をクリア
                avg_reward = avg_reward_temp / 100
                avg_step = avg_step_temp / 100 # Q学習の元のコードでは達成エピソード数で割っていたが、ここでは元のコードに合わせる
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.2f}{RESET}\n")
                avg_reward_temp, avg_step_temp = 0, 0

            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            states = self.env.reset()

            done = False
            step_count = 0
            ep_reward = 0
            losses = [] # エピソードごとの損失を格納

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions = []
                for i, agent in enumerate(self.agents):
                    # Agentクラスのdecay_epsilon_powを呼び出し
                    agent.decay_epsilon_pow(total_step)

                    # Agentクラスのget_actionを呼び出し (global_stateのみ渡す)
                    actions.append(agent.get_action(states))

                # エージェントの状態を保存（オプション）
                agent_positions_in_states = self.env.get_agent_positions(states) # GridWorldのメソッドを使用
                for i, pos in enumerate(agent_positions_in_states):
                    # saver インスタンスは基底クラスで定義したが、派生クラスで初期化済み
                    self.saver.log_agent_states(episode_num, step_count, i, pos)


                # 環境にステップを与えて状態を更新
                next_state, reward, done = self.env.step(states, actions)

                # Q学習は経験ごとに逐次更新
                step_losses = [] # 各ステップでのエージェントごとの損失
                if self.load_model == 0: # 学習モードの場合のみ
                    for i, agent in enumerate(self.agents):
                        # Agentクラスのlearnメソッドを呼び出し (global_state, action, reward, next_global_state, doneを渡す)
                        loss = agent.learn(states, actions[i], reward, next_state, done)
                        step_losses.append(loss)

                # ステップの平均損失をエピソード損失リストに追加
                avg_step_loss = sum(step_losses) / len(step_losses) if step_losses else 0
                losses.append(avg_step_loss)

                states = next_state # 状態を更新
                ep_reward += reward

                step_count += 1
                total_step += 1

            # エピソード終了
            # エピソードの平均損失を計算
            avg_episode_loss = sum(losses) / len(losses) if losses else 0

            # ログにスコアを記録
            # saver インスタンスは基底クラスで定義したが、派生クラスで初期化済み
            self.saver.log_scores(episode_num, step_count, ep_reward, avg_episode_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行


    # MultiAgentBase の save_results 抽象メソッドを実装
    def save_results(self):
        """
        Qテーブルを保存する.
        """
        # saver インスタンスは基底クラスで定義したが、派生クラスで初期化済み
        print("Saving Q-Tables for each agent...")
        # Agentクラスのsave_q_tableメソッドを呼び出す
        for i, agent in enumerate(self.agents):
            agent.save_q_table() # Agentは自身の保存パスを知っている

    # result_show メソッドは基底クラスに実装されているため、ここでは不要
    # 必要であればオーバーライドも可能ですが、今回は基底クラスの実装を使います