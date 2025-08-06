import sys
import os

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

from Q_learn.Agent_Q import Agent
from Enviroments.MultiAgentGridEnv import MultiAgentGridEnv
from utils.Model_Saver import Saver
from utils.plot_results import PlotResults

class MultiAgent_Q:
    def __init__(self, args, agents:list[Agent]): # Expects a list of Agent instances
        # Use the MultiAgentGridEnv class defined in previous cells
        self.env = MultiAgentGridEnv(args)
        self.agents = agents # Store the list of Agent instances

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_number = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number
        self.grid_size = args.grid_size

        self.load_model = args.load_model
        # self.mask = args.mask # Remove this line

        # Save directory calculation remains the same
        save_dir = os.path.join(
            "output",
            f"Q_Reward[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_number}]_goals[{self.goals_number}]"
        )
        # argsにdir_path属性を追加 (Agentクラスの初期化で使用される)
        args.dir_path = save_dir

        self.saver = Saver(save_dir=save_dir,grid_size=self.grid_size)
        self.plot_results = PlotResults(save_dir)


    def run(self):

        # 事前条件チェック
        if self.agents_number > self.goals_number:
            print('goals_num >= agents_num に設定してください.\n')
            sys.exit()

        # 学習開始メッセージ
        # IQL/CQLの表示はAgentクラスの実装に依存するが、ここではMultiAgent_QがQ学習を orchestrate していることを示す
        # if self.mask: # Remove this if condition
        #     # Note: The mask logic might need to be handled within the Agent's learn method
        #     print(f"{GREEN}IQL/CQL (Q学習ベース) で学習中{RESET}\n")
        # else:
        print(f"{GREEN}Q学習で学習中{RESET}\n")


        total_step = 0
        avg_reward_temp, avg_step_temp = 0, 0

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_number + 1):
            if episode_num % 10 == 0:
                print(f"Episode {episode_num} / {self.episode_number}", end='\r', flush=True)


            # 100エピソードごとに平均を出力
            if episode_num % 100 == 0:
                print() # 改行して進捗表示をクリア
                avg_reward = avg_reward_temp / 100
                avg_step = avg_step_temp / 100
                # エピソードごとの平均損失も計算し、表示に追加
                avg_loss = sum(losses) / len(losses) if losses else 0 # ここで losses は過去100エピソードの平均損失リスト
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 loss   : {GREEN}{avg_loss:.4f}{RESET}\n")
                avg_reward_temp, avg_step_temp = 0, 0
                losses = [] # 100エピソードごとに損失リストもリセット
                
            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # The reset method in the updated MultiAgentGridEnv now returns observation, info
            current_states:tuple[tuple[int, int], ...] = self.env.reset()

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
                    actions.append(agent.get_action(current_states))

                # エージェントの状態を保存（オプション）
                # states はゴール位置 + エージェント位置のタプルになっている
                # エージェントの位置は states の self.goals_num 以降
                # TODO: GridWorldのget_agent_positionsメソッドの使用を検討
                agent_positions_dict = self.env.get_agent_positions() # GridWorldのメソッドを使用
                # The get_agent_positions method returns a dictionary {agent_id: (x, y)}
                # Need to iterate through agent_ids to get positions in order
                agent_positions_list = [agent_positions_dict[agent_id] for agent_id in self.env._agent_ids]


                for i, pos in enumerate(agent_positions_list):
                    # Saver expects agent_idx, position, step
                    self.saver.log_agent_states(i, pos[0], pos[1]) # Use step_count here


                # 環境にステップを与えて状態を更新
                # The step method in the updated MultiAgentGridEnv now returns observation, reward, done, info
                next_observation, reward, done, info = self.env.step(actions)

                # Q学習は経験ごとに逐次更新
                step_losses = [] # 各ステップでのエージェントごとの損失
                if self.load_model == 0: # 学習モードの場合のみ
                    for i, agent in enumerate(self.agents):
                        # if self.mask == 0: # Remove this inner if condition if it exists (checked in instruction 4)
                        # Agentクラスのlearnメソッドを呼び出し (global_state, action, reward, next_global_state, doneを渡す)
                        # Note: In a multi-agent setting, the reward might be individual or shared.
                        # The current implementation passes the single global reward to all agents.
                        # This might need adjustment based on the specific MARL algorithm (e.g., IQL).
                        loss = agent.learn(current_states, actions[i], reward, next_observation, done)
                        step_losses.append(loss)

                # ステップの平均損失をエピソード損失リストに追加
                avg_step_loss = sum(step_losses) / len(step_losses) if step_losses else 0
                losses.append(avg_step_loss)


                current_states = next_observation # 状態を更新
                ep_reward += reward

                step_count += 1
                total_step += 1

            # エピソード終了
            # エピソードの平均損失を計算
            avg_episode_loss = sum(losses) / len(losses) if losses else 0


            # ログにスコアを記録
            self.saver.log_episode_data(episode_num, step_count, ep_reward, avg_episode_loss)

            avg_reward_temp += ep_reward
            avg_step_temp += step_count

        self.saver.save_remaining_episode_data()
        self.saver.save_visited_coordinates()
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

    def result_save(self):
        print("Saving results...")
        self.plot_results.draw()
        self.plot_results.draw_heatmap()
