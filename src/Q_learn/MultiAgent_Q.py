import sys
import os
from typing import Dict

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


from Environments.MultiAgentGridEnv import MultiAgentGridEnv
from utils.Saver import Saver
from utils.plot_results import PlotResults
from .IO_Handler import IOHandler
from utils.render import Render

from .Agent_Q import Agent
from .QTable import QTableType

from Base.Constant import PosType

class MultiAgent_Q:
    def __init__(self, args, agents:list[Agent]): # Expects a list of Agent instances
        self.agents = agents # Store the list of Agent instances

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_number = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number
        self.grid_size = args.grid_size

        self.agents_ids = [f"agent_{id}" for id in range(self.agents_number)]
        self.goals_ids = [f"goal_{id}" for id in range(self.goals_number)]

        folder_name = f"Qテーブル_報酬[{self.reward_mode}]_[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_A[{self.agents_number}]_G[{self.goals_number}]"

        self.save_dir = os.path.join(
            "output",
            folder_name
        )

        self._start_episode = 1

        goal_pos_list = [(args.grid_size-1, args.grid_size-1),(args.grid_size//2, args.grid_size//4),(args.grid_size-1, args.grid_size//3), (0, args.grid_size//2)]
        self.env = MultiAgentGridEnv(args, goal_pos_list[:self.goals_number])# 環境クラス. 

        self.goal_pos = tuple(self.env.get_goal_positions().values())

        print("pos", self.goal_pos)
        print("Qtable-len: ", self.agents[0].get_q_table_size())

        # 学習で発生したデータを保存するクラス
        os.makedirs(self.save_dir,exist_ok=True)
        score_sammary_path = os.path.join(self.save_dir, "aggregated_episode_metrics_10.csv")
        visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")

        self.saver = Saver(score_sammary_path,visited_coordinates_path,self.grid_size)
        self.saver.CALCULATION_PERIOD = 10

        # saverクラスで保存されたデータを使ってグラフを作るクラス
        self.plot_results = PlotResults(score_sammary_path,visited_coordinates_path)

    def save_checkpoint(self, episode:int, goal_position:tuple[PosType,...]|list[PosType]):
        """
        保持している各AgentのQテーブルと学習状態をチェックポイントファイルに保存する。
        IOHandlerクラスを使用して保存処理を行う。
        """
        print("チェックポイント保存中...")
        io_handler = IOHandler()
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        temp_path_list = []
        for id, agent in enumerate(self.agents):
            q_table_data = agent.get_weights()
            temp_path = os.path.join(checkpoint_dir, f'temp_agent_{id}_checkpoint.pth') # ファイル名を変更
            temp_path_list.append(temp_path)
            save_param = {"state_dict":q_table_data, "episode": episode, "goal_position": goal_position, "epsilon": agent.epsilon}
            io_handler.save(save_param, temp_path)

        for id,agent in enumerate(self.agents):
            file_path = os.path.join(checkpoint_dir, f'agent_{id}_checkpoint.pth')
            if os.path.exists(file_path): os.remove(file_path)# ファイルがすでにあればリネームと競合しないように消す
            os.rename(temp_path_list[id],file_path)

    def save_model(self):
        """
        保持している各AgentのQテーブルをモデルファイルに保存する。
        IOHandlerクラスを使用して保存処理を行う。
        """
        print("モデル保存中...")
        io_handler = IOHandler()
        model_dir = os.path.join(self.save_dir, "models") # ファイルパス生成方法を修正
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for id, agent in enumerate(self.agents):
            q_table_data = agent.get_weights()
            file_path = os.path.join(model_dir, f'agent_{id}_model.pth') # ファイル名を変更
            io_handler.save(q_table_data, file_path)

    def load_checkpoint(self)->list[PosType]:
        """
        ファイルから各AgentのQテーブルと学習状態を読み込み、対応するAgentに設定する。
        IOHandlerクラスを使用して読み込み処理を行う。
        """
        print("チェックポイント読み込み中...")
        io_handler = IOHandler()
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        goal_pos:list[PosType] = [] # 読み込まれた、または新規のゴール位置を格納

        all_checkpoints_found = True # 全てのエージェントのチェックポイントが見つかったかを示すフラグ
        loaded_episode = 0 # 読み込まれたエピソード数
        loaded_epsilon = 1.0

        for id, agent in enumerate(self.agents):
            file_path = os.path.join(checkpoint_dir, f'agent_{id}_checkpoint.pth') # ファイル名を変更
            load_data:dict = io_handler.load(file_path)

            if load_data: # データが読み込まれた場合
                qtable: QTableType = load_data.get('state_dict', {}) # 存在しないキーの場合に備えてgetを使用
                agent.set_weights(qtable)

                # 最初の有効なチェックポイントからエピソード数とゴール位置を取得
                if not goal_pos: # goal_posがまだ設定されていない場合
                    goal_pos = load_data.get('goal_position', [])
                if loaded_episode == 0: # loaded_episodeがまだ設定されていない場合
                    loaded_episode = load_data.get('episode', 0)
                if loaded_epsilon == 1.0:
                    loaded_epsilon = load_data.get('epsilon', 1.0)

            else: # チェックポイントファイルが存在しない、または読み込みエラーの場合
                print(f"エージェント {id} のチェックポイントが見つからないか、読み込みに失敗しました。Qテーブルを初期化します。")
                agent.set_weights({}) # AgentクラスのQテーブル初期化メソッドを呼び出す
                all_checkpoints_found = False # 1つでも見つからなければFalse

        if all_checkpoints_found:
            self._start_episode = loaded_episode

            # εを設定
            for agent in self.agents:
                agent.epsilon = loaded_epsilon

            return goal_pos # 読み込まれたゴール位置を返す
        else:
            # 1つでもチェックポイントが見つからなかった場合、新規学習として扱う
            self._start_episode = 1
            # 新規の場合のゴール位置サンプリングは__init__で行われるため、ここでは特別な処理は不要
            return [] # 空のリストを返すことで、__init__で新規サンプリングを促す

    def load_model(self):
        """
        ファイルから各AgentのQテーブルを読み込み、対応するAgentに設定する (推論用など)。
        IOHandlerクラスを使用して読み込み処理を行う。
        """
        print("モデル読み込み中...")

        io_handler = IOHandler()
        for id, agent in enumerate(self.agents):
            file_path = os.path.join(self.save_dir, "models", f'agent_{id}_model.pth') # ファイル名を変更

            qtable: QTableType = io_handler.load(file_path)
            if qtable: # 読み込みに成功した場合のみ設定
                agent.set_weights(qtable)
            else:
                print(f"エージェント {id} のモデルが見つからないか、読み込みに失敗しました。Qテーブルは更新されません。")

    def result_save(self):
        print("Saving results...")
        self.plot_results.draw()
        self.plot_results.draw_heatmap()

    def make_trajectry(self):
        trajectory_data_for_render = []
        done = False
        time_step = 0
        total_episode_reward = 0.0
        action_str = ["↑","↓","←","→","停"]

        # Reset environment to get initial agent observations
        # Use the same initial positions as in train, or let it random if train also random.
        # Here I'll use the same fixed initial positions as the `train` method uses for agent placement.
        initial_agent_positions_for_reset = [(0,i) for i in range(self.agents_number)]
        agent_observations = self.env.reset(initial_agent_positions=initial_agent_positions_for_reset)

        # Fixed goal positions (assuming they don't change during an episode)
        fixed_goal_positions = list(self.env.get_goal_positions().values())

        # Initial agent positions from the observation
        current_agent_positions_list = [agent_observations[agent_id]['self'] for agent_id in self.agents_ids]
        trajectory_data_for_render.append(fixed_goal_positions + current_agent_positions_list)

        for agent in self.agents:
            agent.epsilon = 0.0 # Force exploitation during trajectory generation

        while not done and time_step < 50:
            actions: Dict[str, int] = {}
            for agent_idx, agent in enumerate(self.agents):
                agent_id = self.agents_ids[agent_idx]
                action = agent.get_action(agent_observations)
                # Print agent's action and Q-values for debugging
                # Format Q-values to 3 decimal places and add color for max/min
                q_values_raw = agent.q_table.get_q_values(agent._get_q_state(agent_observations))
                
                if q_values_raw:
                    max_q = max(q_values_raw)
                    min_q = min(q_values_raw)
                    colored_q_values = []
                    for q in q_values_raw:
                        if q == max_q:
                            colored_q_values.append(f"{GREEN}{q:.3f}{RESET}")
                        elif q == min_q:
                            colored_q_values.append(f"{RED}{q:.3f}{RESET}")
                        else:
                            colored_q_values.append(f"{q:.3f}")
                    formatted_q_values_str = '[' + ', '.join(colored_q_values) + ']'
                else:
                    formatted_q_values_str = '[]'
                
                print(f"[{time_step}] agent {agent.id} 行動{GREEN}[{action_str[action]}]{RESET}: {formatted_q_values_str}")
                actions[agent_id] = action

            next_agent_observations, reward_dict, done_dict, _ = self.env.step(actions)
            done = done_dict["__all__"]

            # Extract next agent positions for trajectory logging
            next_agent_positions_list = [next_agent_observations[agent_id]['self'] for agent_id in self.agents_ids]
            trajectory_data_for_render.append(fixed_goal_positions + next_agent_positions_list)

            # Update for next step
            agent_observations = next_agent_observations
            total_episode_reward += sum(reward_dict.values())
            # Print current step information (state transition and reward for this step)
            # The previous print was just printing the same `states` and `reward` variables which were not updated in the loop.
            print(f"state at step {time_step}: {agent_observations} \nnext_state at step {time_step}: {next_agent_observations}\nreward for this step: {sum(reward_dict.values())}")
            time_step += 1

        print("Total reward for trajectory: ", total_episode_reward)
        return trajectory_data_for_render, total_episode_reward, done

    def render_anime(self, total_episode):
        traj, r, done = self.make_trajectry()
        print(f"Trajectory reward: {r}, done: {done}") # Updated print
        render = Render(self.grid_size, self.goals_number, self.agents_number)
        render.render_anime(traj, os.path.join(self.save_dir, f"{total_episode}.gif"))

    def train(self,total_episodes:int):
        # 事前条件チェック
        if self.agents_number > self.goals_number:
            print('goals_num >= agents_num に設定してください。\n')
            sys.exit()

        if total_episodes <= self._start_episode:
            print(f"学習されない: {total_episodes} <= {self._start_episode}")
            return

        # 学習開始メッセージ
        print(f"{GREEN}Q学習で学習中{RESET}\n")
        print(f"ゴール位置: {self.env.get_goal_positions().values()}")

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        print(f"---------学習開始 (全 {total_episodes} エピソード)----------")

        episode_losses = []
        episode_steps  = []
        episode_rewards= []
        episode_dones  = []

        for episode in range(self._start_episode, total_episodes + 1):
            if episode % 10 == 0:
                print(f"Episode {episode} / {total_episodes}", end='\r', flush=True)

            if episode % 1000 == 0:
                self.save_checkpoint(episode, self.goal_pos)

            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # The reset method in the updated MultiAgentGridEnv now returns observation, info
            init_state = [(0,i) for i in range(self.agents_number)]
            current_observations:Dict = self.env.reset(initial_agent_positions=init_state)

            episode_done_overall = False # エピソード全体の終了フラグ
            step_count = 0
            episode_reward:float = 0.0
            step_losses:list[float] = [] # 各ステップでのエージェントごとの損失

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not episode_done_overall and step_count < self.max_ts:
                act: list = []
                actions:dict[str, int] = {}
                for i, agent in enumerate(self.agents):
                    # Agentクラスのdecay_epsilon_powを呼び出し
                    agent.decay_epsilon_power()

                    # Agentクラスのget_actionを呼び出し (global_stateのみ渡す)
                    act.append(agent.get_action(current_observations))

                for i, aid in enumerate(self.agents_ids):
                    actions[aid] = act[i]

                # エージェントの状態を保存（オプション）
                agent_positions_dict = self.env.get_agent_positions() # GridWorldのメソッドを使用
                agent_positions_list = [agent_positions_dict[agent_id] for agent_id in self.env._agent_ids]

                for i, pos in enumerate(agent_positions_list):
                    # Saver expects agent_idx, position, step
                    self.saver.log_agent_states(i, pos[0], pos[1]) # Use step_count here


                # 環境にステップを与えて状態を更新
                next_observations, reward_dict, done_dict, info = self.env.step(actions)
                episode_done_overall = done_dict["__all__"]

                # Q学習は経験ごとに逐次更新
                # 各エージェントに対して学習を実行
                for aid, agent in enumerate(self.agents):
                    agent_id = self.agents_ids[aid]
                    # 各エージェントの observe と learn には、そのエージェントに紐づく報酬と終了フラグを渡す
                    agent.observe(
                        current_observations,
                        int(actions[agent_id]),
                        float(reward_dict[agent_id]),
                        next_observations,
                        bool(done_dict[agent_id]) # 各エージェントのdone状態を渡す
                    )
                    loss = agent.learn()
                    step_losses.append(loss)

                current_observations = next_observations # 状態を更新
                episode_reward += sum(reward_dict.values())
                step_count += 1

            # エピソード終了
            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses) if step_losses else 0.0
            # エージェントのQテーブルサイズを取得
            q_table_size_list = [agent.get_q_table_size() for agent in self.agents]

            # ログにスコアを記録
            # self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, episode_done_overall, q_table_size_list)
            self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, episode_done_overall)

            episode_steps.append(step_count)
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)
            episode_dones.append(episode_done_overall)

            # 暫定的にここで情報集計
            if episode % 50 == 0 and episode > 0:
                mean = lambda arr: float(sum(arr)/len(arr))
                print(f"\n\t エピソード {episode-49} ~ {episode} の平均 step  : \033[92m{mean(episode_steps):.2f}\033[0m\t{GREEN}{min(episode_steps)}{RESET}/{RED}{max(episode_steps)}{RESET}")
                print(  f"\t エピソード {episode-49} ~ {episode} の平均 reward: \033[92m{mean(episode_rewards):.2f}\033[0m\t{GREEN}{max(episode_rewards)}{RESET}/{RED}{min(episode_rewards)}{RESET}")
                print(  f"\t エピソード {episode-49} ~ {episode} の平均 loss  : \033[92m{mean(episode_losses):.4f}\033[0m\t{GREEN}{min(episode_losses):.4f}{RESET}/{RED}{max(episode_losses):.4f}{RESET}")
                print(  f"\t エピソード {episode-49} ~ {episode} の成功率     : \033[92m{mean(episode_dones):.3f}\033[0m")
                print(  f"\t データ量   : \033[92m{q_table_size_list}\033[0m, 探索率 : {self.agents[0].epsilon:.2f}, 各データ{len(episode_losses)},{len(episode_rewards)}\n")

                episode_losses = []
                episode_steps  = []
                episode_rewards= []
                episode_dones  = []

        self.saver.save_visited_coordinates()
        self.save_model()
        print(f"---------学習終了 (全 {total_episodes} エピソード完了)----------")
