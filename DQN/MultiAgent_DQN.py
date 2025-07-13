import sys
import torch
import os
import csv
import numpy as np

from env import GridWorld
from DQN.Agent_DQN import Agent_DQN
from utils.plot_results import PlotResults
from utils.Model_Saver import Saver


RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Modify MultiAgent_DQN.run to pass total_episode_num to learn_from_experience
class MultiAgent_DQN:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, agents:list[Agent_DQN]):
        """
        MultiAgent_DQN クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_num, grid_size, load_model, mask,
                   save_agent_states, alpha, beta, beta_anneal_steps 属性を持つことを想定)
            agents (list[Agent_DQN]): 使用するエージェントオブジェクトのリスト.
        """
        self.env = GridWorld(args)
        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size
        self.load_model = args.load_model
        self.mask = args.mask

        self.save_agent_states = args.save_agent_states

        # 結果保存ディレクトリの設定と作成
        save_dir = os.path.join(
            "output",
            f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]"
            # PERを使用する場合、ディレクトリ名にPER関連パラメータを含めるとより分かりやすい
            # f"DQN_PER_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]_alpha[{args.alpha}]_beta_anneal[{args.beta_anneal_steps}]"
        )
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        # 結果保存およびプロット関連クラスの初期化
        self.saver = Saver(save_dir)
        self.plot_results = PlotResults(save_dir)


    def run(self):
        """
        強化学習のメイン実行ループ.
        指定されたエピソード数だけ環境とのインタラクションと学習を行います。
        """
        # 事前条件チェック: ゴール数はエージェント数以下である必要がある
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}DQN{RESET} で学習中...\n")

        total_step = 0 # 環境との全インタラクションステップ数の累積
        # 集計用一時変数の初期化
        avg_reward_temp = 0
        avg_step_temp = 0
        achieved_episodes_temp = 0
        avg_loss_temp = 0
        learning_steps_in_period = 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            # 100エピソードごとの集計期間の開始時に変数をリセット
            if (episode_num - 1) % 100 == 0:
                avg_reward_temp = 0
                avg_step_temp = 0
                achieved_episodes_temp = 0
                avg_loss_temp = 0
                learning_steps_in_period = 0

            print('■', end='',flush=True)  # 進捗表示 (エピソード100回ごとに改行)

            # 100エピソードごとに集計結果を出力
            if episode_num % 100 == 0:
                print() # 改行
                avg_reward = avg_reward_temp / 100       # 期間内の平均報酬

                avg_step = 0.00     # 期間内の平均ステップ数
                if achieved_episodes_temp > 0: # 達成したエピソードがある場合のみ平均ステップ数を計算
                    avg_step = avg_step_temp / achieved_episodes_temp

                achievement_rate = achieved_episodes_temp / 100    # 達成率を計算 (達成したエピソード数 / 集計エピソード数)

                # 平均損失は学習が発生したステップ数で割る
                avg_loss = avg_loss_temp / learning_steps_in_period if learning_steps_in_period > 0 else 0

                print(f"     エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.3f}{RESET}")
                print(f"     エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.3f}{RESET}")
                print(f"     エピソード {episode_num - 99} ~ {episode_num} の達成率     : {GREEN}{achievement_rate:.2f}{RESET}") # 達成率も出力 .2f で小数点以下2桁表示
                print(f"     エピソード {episode_num - 99} ~ {episode_num} の平均 loss  : {GREEN}{avg_loss:.5f}{RESET}\n") # 平均損失も出力


            # 各エピソード開始時に環境をリセット
            current_global_state = self.env.reset()

            done = False # エピソード完了フラグ
            step_count = 0 # 現在のエピソードのステップ数
            ep_reward = 0.0 # 現在のエピソードの累積報酬

            # Variables to track loss for this episode's logging
            ep_total_loss = 0
            ep_learning_steps = 0

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                # 各エージェントの行動を選択
                actions = []
                for i, agent in enumerate(self.agents):
                    # エージェントにε減衰を適用 (全ステップ数に基づき減衰)
                    agent.decay_epsilon_power(total_step)

                    # エージェントに行動を選択させる
                    # エージェント内部で自身の観測(masking)を行うため、全体状態を渡す
                    actions.append(agent.get_action(i, current_global_state))

                # エージェントの状態を保存（オプション）
                # 全体状態からエージェント部分を抽出 し、Saverでログ記録
                if self.save_agent_states:
                     # current_global_state の構造に依存してエージェント部分を抽出
                     # モックの構造に合わせて、agent_pos は goals_num 以降とする
                    agent_positions_in_global_state = current_global_state[self.goals_num:]
                    for i, agent_pos in enumerate(agent_positions_in_global_state):
                        self.saver.log_agent_states(episode_num, step_count, i, agent_pos)

                # 環境にステップを与えて状態を更新し、結果を取得
                # 入力に現在の全体状態と全エージェントの行動を使用
                next_global_state, reward, done = self.env.step(current_global_state, actions)

                # 各ステップで獲得した報酬をエピソード報酬に加算
                ep_reward += reward

                # 各エージェントの経験をリプレイバッファにストアし、学習を試行
                for i, agent in enumerate(self.agents):
                    # エージェントは自身の経験 (状態s, 行動a, 報酬r, 次状態s', 終了フラグdone) をストア
                    # 状態sと次状態s'は環境全体の全体状態を渡す
                    agent.observe_and_store_experience(current_global_state, actions[i], reward, next_global_state, done)

                    # エージェントに学習を試行させる (総エピソード数を渡す) (Step 4)
                    # learn_from_experience はバッファサイズが満たされているなど、学習可能な場合に損失を返す
                    current_loss = agent.learn_from_experience(i, episode_num, self.episode_num) # 総エピソード数を渡す
                    if current_loss is not None:
                        # 学習が発生した場合、その損失を累積 (for episode logging)
                        ep_total_loss += current_loss
                        ep_learning_steps += 1 # Count steps where at least one agent learned in this episode
                        # Also accumulate for the 100-episode period average
                        avg_loss_temp += current_loss
                        learning_steps_in_period += 1 # Count learning steps in the period


                # 全体状態を次の状態に更新
                current_global_state = next_global_state

                step_count += 1 # エピソード内のステップ数をインクリメント
                total_step += 1 # 全体のステップ数をインクリメント

            # ---------------------------
            # エピソード終了後の処理
            # ---------------------------

            # エピソードが完了 (done == True) した場 合、達成エピソード数カウンタをインクリメント
            if done:
                achieved_episodes_temp += 1
                avg_step_temp += step_count # 達成した場合のステップ数のみ加算

            # Calculate average loss for the episode for logging
            ep_avg_loss = ep_total_loss / ep_learning_steps if ep_learning_steps > 0 else 0

            # Saverでエピソードごとのスコアをログに記録
            # エピソード番号、最終ステップ数、累積報酬、エピソード中の平均損失を記録
            self.saver.log_scores(episode_num, step_count, ep_reward, ep_avg_loss)

            # 集計期間内の平均計算のための累積 (avg_reward_temp accumulation)
            avg_reward_temp += ep_reward


        print()  # 全エピソード終了後に改行

    def save_model_weights(self):
        """学習済みモデルの重みを保存する."""
        # モデル保存を Saver に依頼
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
        # PlotResults にプロットを依頼
        self.plot_results.draw()
        self.plot_results.draw_heatmap(self.grid_size)
