import sys
import os

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

from Q_learn.Agent_Q import Agent
from env import GridWorld
from utils.Model_Saver import Saver
from utils.plot_results import PlotResults

class MultiAgent_Q:
    def __init__(self, args, agents:list[Agent]): # Expects a list of Agent instances
        self.env = GridWorld(args)
        self.agents = agents # Store the list of Agent instances

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size

        self.load_model = args.load_model
        self.mask = args.mask

        # Save directory calculation remains the same
        save_dir = os.path.join(
            "output",
            f"Q_mask[{self.mask}]_Reward[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_num}]"
        )
        # argsにdir_path属性を追加 (Agentクラスの初期化で使用される)
        args.dir_path = save_dir

        # Mock SaverとMock PlotResultsを使用
        self.saver = Saver(save_dir=save_dir)
        self.plot_results = PlotResults(save_dir)


    def run(self):

        # 事前条件チェック
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        # IQL/CQLの表示はAgentクラスの実装に依存するが、ここではMultiAgent_QがQ学習を orchestrate していることを示す
        if self.mask:
            # Note: The mask logic might need to be handled within the Agent's learn method
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
                avg_step = avg_step_temp / 100
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
                # states はゴール位置 + エージェント位置のタプルになっている
                # エージェントの位置は states の self.goals_num 以降
                # TODO: GridWorldのget_agent_positionsメソッドの使用を検討 (MockGridWorldには実装済み)
                agent_positions_in_states = self.env.get_agent_positions(states) # GridWorldのメソッドを使用
                for i, pos in enumerate(agent_positions_in_states):
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
            self.saver.log_scores(episode_num, step_count, ep_reward, avg_episode_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        print()  # 終了時に改行

    def save_model_weights(self):
        # モデル保存やプロット - Q学習ではQテーブルを保存するのでこのメソッドは使わない
        print("Mock save_model_weights called - Not applicable for Q-Learning.")
        pass # DQNではないため何もしない

    def save_Qtable(self):
        print("Saving Q-Tables for each agent...")
        # Agentクラスのsave_q_tableメソッドを呼び出す
        for i, agent in enumerate(self.agents):
            agent.save_q_table() # Agentは自身の保存パスを知っている

    def result_show(self):
        print("Showing results...")
        self.plot_results.draw()
        self.plot_results.draw_heatmap(self.grid_size)