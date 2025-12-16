import sys
import os
from typing import Optional # Added for type hinting

from Enviroments.MultiAgentGridEnv import MultiAgentGridEnv

from utils.plot_results import PlotResults
from utils.Saver import Saver

from .Agent_DQN import Agent
from .IO_Handler import Model_IO
# from .dqn import QNet

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# Modify MultiAgent_DQN.run to pass total_episode_num to learn_from_experience
class MultiAgent_DQN:
    """
    複数のDQNエージェントを用いた強化学習の実行を管理するクラス.
    環境とのインタラクション、エピソードの進行、学習ループ、結果の保存・表示を統括します。
    """
    def __init__(self, args, agents:list[Agent]):
        """
        MultiAgent_DQN クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
                  (reward_mode, render_mode, episode_number, max_timestep,
                   agents_number, goals_num, grid_size, load_model, mask,
                   save_agent_states, alpha, beta, beta_anneal_steps, use_per 属性を持つことを想定)
            agents (list[Agent_DQN]): 使用するエージェントオブジェクトのリスト.
        """
        self.env = MultiAgentGridEnv(args, fixrd_goals=[(args.grid_size-1,args.grid_size-1),(args.grid_size//2,2*args.grid_size//3)])
        # self.env = MultiAgentGridEnv(args, fixrd_goals=[(args.grid_size-1,args.grid_size-1)])
        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number # Fixed: Changed args.goals_num to args.goals_number
        self.grid_size = args.grid_size
        self.load_model = args.load_model
        self.mask = args.mask
        self.update_frequency = 4# 学習の頻度

        self.save_agent_states = args.save_agent_states

        # 結果保存ディレクトリの設定と作成
        self.save_dir = os.path.join(
            "output",
            f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]" + (f"_PER_alpha[{args.alpha}]_beta_anneal[{args.beta_anneal_steps}]" if args.use_per else "")
        )

        # cp_dir = os.path.join(self.save_dir, ".checkpoints")
        # json_data = {"agents_number":self.agents_number,"goal": {"number":self.goals_number,"position":self.env.get_goal_positions()}}
        # conf = ConfigManager(json_data, cp_dir)   # 設定ファイルをフォルダに作成
        # print("conf.get_setting('agents_number'):",conf.get_setting("agents_number"))

        # 結果保存およびプロット関連クラスの初期化
        self.saver = Saver(self.save_dir,self.grid_size)
        self.plot_results = PlotResults(self.save_dir)

    def result_save(self):
        self.plot_results.draw_heatmap()
        self.plot_results.draw()

    def train(self):
        """
        強化学習のメイン実行ループ.
        指定されたエピソード数だけ環境とのインタラクションと学習を行います。
        """
        # 事前条件チェック: エージェント数はゴール数以下である必要がある
        if self.agents_number > self.goals_number:
            print('goals_num >= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        print(f"{GREEN}DQN{RESET} で学習中..." + (f" ({GREEN}PER enabled{RESET})" if self.agents[0].use_per else "") + "\n")
        print(f"goals: {self.env.get_goal_positions().values()}")

        total_step = 0 # 環境との全インタラクションステップ数の累積
        # 集計用一時変数の初期化
        avg_reward_temp = 0
        avg_step_temp = 0
        achieved_episodes_temp = 0
        avg_loss_temp = 0
        learning_steps_in_period = 0
        renzoku_not_done = 0# 連敗記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_num + 1):
            # 100エピソードごとの集計期間の開始時に変数をリセット
            if (episode_num - 1) % 50 == 0:
                avg_reward_temp = 0
                avg_step_temp = 0
                achieved_episodes_temp = 0
                avg_loss_temp = 0
                learning_steps_in_period = 0

            if renzoku_not_done % 100==0 and renzoku_not_done>0:# 連敗が続くとき探索を強制
                pre_epsilon = self.agents[0].epsilon
                for agent in self.agents:
                    agent.epsilon = min(agent.epsilon+0.50, 1.0)
                print(f"連敗記録:{renzoku_not_done}。探索率を{pre_epsilon:.3}から{agent.epsilon:.3}に上昇。")

            print('■', end='',flush=True)  # 進捗表示 (エピソード100回ごとに改行)

            # 100エピソードごとに集計結果を出力
            CONSOLE_LOG_FREQ = 50
            if episode_num % CONSOLE_LOG_FREQ == 0:
                print() # 改行
                avg_reward = avg_reward_temp / CONSOLE_LOG_FREQ       # 期間内の平均報酬

                avg_step = 0.00     # 期間内の平均ステップ数
                if achieved_episodes_temp > 0: # 達成したエピソードがある場合のみ平均ステップ数を計算
                    avg_step = avg_step_temp / achieved_episodes_temp

                achievement_rate = achieved_episodes_temp / CONSOLE_LOG_FREQ    # 達成率を計算 (達成したエピソード数 / 集計エピソード数)

                # 平均損失は学習が発生したステップ数で割る
                avg_loss = avg_loss_temp / learning_steps_in_period if learning_steps_in_period > 0 else 0

                print(f"     エピソード {episode_num - CONSOLE_LOG_FREQ} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.3f}{RESET}")
                print(f"     エピソード {episode_num - CONSOLE_LOG_FREQ} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.3f}{RESET}")
                print(f"     エピソード {episode_num - CONSOLE_LOG_FREQ} ~ {episode_num} の達成率     : {GREEN}{achievement_rate:.2f}{RESET}") # 達成率も出力 .2f で小数点以下2桁表示
                print(f"     エピソード {episode_num - CONSOLE_LOG_FREQ} ~ {episode_num} の平均 loss  : {GREEN}{avg_loss:.5f}{RESET}") # 平均損失も出力
                print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.agents[0].epsilon:.3f}{RESET}, beta: {GREEN}{self.agents[0].beta:.3f}{RESET}") #


            # 各エピソード開始時に環境をリセット
            current_global_state = self.env.reset(initial_agent_positions=[(0,0),(0,1)])
            # current_global_state = self.env.reset(initial_agent_positions=[(0,0)])

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
                # 全体状態からエージェント部分を抽出し、Saverでログ記録
                if self.save_agent_states:
                    agent_positions_in_global_state = current_global_state[self.goals_number:]
                    for i, agent_pos in enumerate(agent_positions_in_global_state):
                        self.saver.log_agent_states(i, agent_pos[0], agent_pos[1])

                # 環境にステップを与えて状態を更新し、結果を取得
                # 入力に現在の全体状態と全エージェントの行動を使用
                next_global_state, reward, done, _ = self.env.step(actions)

                # 各ステップで獲得した報酬をエピソード報酬に加算
                ep_reward += reward

                # 各エージェントの経験をリプレイバッファにストアし、学習を試行
                for i, agent in enumerate(self.agents):
                    # エージェントは自身の経験 (状態s, 行動a, 報酬r, 次状態s', 終了フラグdone) をストア
                    # 状態sと次状態s'は環境全体の全体状態を渡す
                    agent.observe_and_store_experience(current_global_state, actions[i], reward, next_global_state, done)

                # 4ステップに1回学習
                if step_count % self.update_frequency==0:
                    for i, agent in enumerate(self.agents):
                        # エージェントに学習を試行させる (総エピソード数を渡す) (Step 4)
                        # learn_from_experience はバッファサイズが満たされているなど、学習可能な場合に損失を返す
                        current_loss = agent.learn_from_experience(i, total_step) # 総エピソード数を渡す
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

            # エピソードが完了 (done == True) した場合、達成エピソード数カウンタをインクリメント
            if done:
                achieved_episodes_temp += 1
                avg_step_temp += step_count # 達成した場合のステップ数のみ加算
                renzoku_not_done = 0 # 連敗をリセット
            else:
                renzoku_not_done += 1 # 連敗カウンタ

            # Calculate average loss for the episode for logging
            ep_avg_loss = ep_total_loss / ep_learning_steps if ep_learning_steps > 0 else 0

            # Saverでエピソードごとのスコアをログに記録
            # エピソード番号、最終ステップ数、累積報酬、エピソード中の平均損失を記録
            self.saver.log_episode_data(episode_num, step_count, ep_reward, ep_avg_loss, done)

            # 集計期間内の平均計算のための累積 (avg_reward_temp accumulation)
            avg_reward_temp += ep_reward

        self.saver.save_remaining_episode_data()
        self.saver.save_visited_coordinates()
        print()  # 全エピソード終了後に改行

    def save_model_weights(self):
        """学習済みモデルの重みを保存する."""
        model_io = Model_IO()
        model_dir = file_path = os.path.join(self.save_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for id, agent in enumerate(self.agents):
            model_weight, _, _ = agent.get_weights()
            file_path = os.path.join(model_dir, f"model_{id}.pth")
            model_io.save(file_path, model_weight)

    # 変更: QNetインスタンスではなく、state_dictを直接返すように変更
    def load_model_weights(self):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "models")

        for id, agent in enumerate(self.agents):
            file_path = os.path.join(model_dir, f"model_{id}.pth")
            loaded_state_dict = model_io.load(file_path) # state_dictをロード

            # Fixed: Directly load state_dict into the agent's existing qnet and qnet_target modules
            agent.model.qnet.load_state_dict(loaded_state_dict)
            agent.model.qnet_target.load_state_dict(loaded_state_dict)

    def simulate_agent_behavior(self, num_simulation_episodes: int = 1, max_simulation_timestep: Optional[int] = None):
        """
        学習済みモデルを使ってエージェントの行動をシミュレーションします。

        Args:
            num_simulation_episodes (int): シミュレーションを実行するエピソード数。
            max_simulation_timestep (Optional[int]): 各シミュレーションエピソードの最大ステップ数。
                                                     Noneの場合、MultiAgent_DQNのmax_tsが使用されます。
        """
        print(f"{GREEN}--- シミュレーション開始 (学習済みモデル使用) ---" + f"{RESET}")
        print(f"シミュレーションエピソード数: {num_simulation_episodes}\n")

        if max_simulation_timestep is None:
            max_simulation_timestep = self.max_ts

        self.load_model_weights()

        for episode_idx in range(1, num_simulation_episodes + 1):
            print(f"--- シミュレーションエピソード {episode_idx} / {num_simulation_episodes} ---")

            # current_global_state = self.env.reset(initial_agent_positions=[(0,0)]) # エージェントの初期位置を設定
            current_global_state = self.env.reset(initial_agent_positions=[(0,0),(0,1)]) # エージェントの初期位置を設定
            done = False
            step_count = 0
            ep_reward = 0.0

            while not done and step_count < max_simulation_timestep:
                print(f"\nステップ {step_count}:")
                print(f"  現在の全体状態: {current_global_state}")

                # 各エージェントの行動とQ値を収集するためのリスト
                agent_step_info = []

                for i, agent in enumerate(self.agents):
                    # 推論モードではε-greedyの活用部分のみを使用
                    # エージェント内部で自身の観測(masking)を行うため、全体状態を渡す
                    # global_state_tensor への変換は nn_greedy_actor 内で行われる
                    # ε-greedyの探索部分を無効化するため、一時的にepsilonを0にする
                    original_epsilon = agent.epsilon
                    agent.epsilon = 0.0

                    action = agent.get_action(i, current_global_state)
                    all_q_values = agent.get_all_q_values(i, current_global_state) # 全Q値を取得

                    agent_step_info.append({
                        'agent_id': i,
                        'action': action,
                        'q_values': all_q_values.tolist() # Q値をリストに変換して保存
                    })

                    # epsilonを元に戻す
                    agent.epsilon = original_epsilon

                # 各エージェントの行動とQ値を出力
                for info in agent_step_info:
                    print(f"  エージェント {info['agent_id']}: 選択された行動: {info['action']}, Q値: {info['q_values']}")

                # 環境にステップを与えて状態を更新し、結果を取得
                # actionsリストはagent_step_infoから再構築
                actions_for_env = [info['action'] for info in agent_step_info]
                next_global_state, reward, done, _ = self.env.step(actions_for_env)
                ep_reward += reward

                print(f"  報酬: {reward:.2f}, 完了: {done}")

                current_global_state = next_global_state
                step_count += 1

                if done:
                    print(f"エピソード {episode_idx} 完了. 最終ステップ: {step_count}, 累積報酬: {ep_reward:.2f}")
                elif step_count == max_simulation_timestep:
                    print(f"エピソード {episode_idx} タイムアウト. 最終ステップ: {step_count}, 累積報酬: {ep_reward:.2f}")

            print("\n" + "-" * 50 + "\n")

        print(f"{GREEN}--- シミュレーション終了 ---" + f"{RESET}")
