import sys
import os

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


from Enviroments.MultiAgentGridEnv import MultiAgentGridEnv
from utils.Saver import Saver
from utils.plot_results import PlotResults
#from utils.ConfigManager import ConfigManager, ConfigLoader
from .IO_Handler import IOHandler
from utils.render import Render

from .Agent_Q import Agent
from .QTable import QTableType

from Enviroments.Grid import PositionType

class MultiAgent_Q:
    def __init__(self, args, agents:list[Agent]): # Expects a list of Agent instances
        # Use the MultiAgentGridEnv class defined in previous cells
        #self.env = MultiAgentGridEnv(args)
        self.agents = agents # Store the list of Agent instances

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_number = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number
        self.grid_size = args.grid_size

        # IQL: maskあり/CQL: maskなし
        Q_Strategy = "IQL" if args.mask else "CQL"
        self.save_dir = os.path.join(
            "output",
            f"{Q_Strategy}_部分観測[{args.neighbor_distance}]_r[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_number}]_goals[{self.goals_number}]"
        )

        # self.io_handler = IOHandler() # IOHandlerのインスタンスを作成

        self._start_episode = 1

        #goal_pos_list:list[PositionType] = self.load_checkpoint()
        goal_pos_list = self.load_checkpoint()
        self.env = MultiAgentGridEnv(args, goal_pos_list)# 環境クラス
        
        self.goal_pos = tuple(self.env.get_goal_positions().values())
        
        print("pos", self.goal_pos)
        print("Qtable-len: ", self.agents[0].get_q_table_size())

        # 学習で発生したデータを保存するクラス
        self.saver = Saver(self.save_dir,self.grid_size)
        
        # saverクラスで保存されたデータを使ってグラフを作るクラス
        self.plot_results = PlotResults(self.save_dir)

    def save_checkpoint(self, episode:int, goal_position:tuple[PositionType,...]|list[PositionType]):
        """
        保持している各AgentのQテーブルと学習状態をチェックポイントファイルに保存する.
        IOHandlerクラスを使用して保存処理を行う.
        """
        print("チェックポイント保存中...")
        io_handler = IOHandler()
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        temp_path_list = []
        for agent in self.agents:
            q_table_data = agent.get_Qtable()
            temp_path = os.path.join(checkpoint_dir, f'temp_agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
            temp_path_list.append(temp_path)
            save_param = {"state_dict":q_table_data, "episode": episode, "goal_position": goal_position, "epsilon": agent.epsilon}
            io_handler.save(save_param, temp_path) # IOHandlerを使用

        for i,agent in enumerate(self.agents):
            file_path = os.path.join(checkpoint_dir, f'agent_{agent.agent_id}_checkpoint.pth')
            if os.path.exists(file_path): os.remove(file_path)# ファイルがすでにあればリネームと競合しないように消す
            os.rename(temp_path_list[i],file_path)

    def save_model(self):
        """
        保持している各AgentのQテーブルをモデルファイルに保存する.
        IOHandlerクラスを使用して保存処理を行う.
        """
        print("モデル保存中...")
        io_handler = IOHandler()
        model_dir = os.path.join(self.save_dir, "models") # ファイルパス生成方法を修正
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for agent in self.agents:
            q_table_data = agent.get_Qtable()
            file_path = os.path.join(model_dir, f'agent_{agent.agent_id}_model.pth') # ファイル名を変更
            io_handler.save(q_table_data, file_path) # IOHandlerを使用

    def load_checkpoint(self)->list[PositionType]:
        """
        ファイルから各AgentのQテーブルと学習状態を読み込み、対応するAgentに設定する.
        IOHandlerクラスを使用して読み込み処理を行う.
        """
        print("チェックポイント読み込み中...")
        io_handler = IOHandler()
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        goal_pos:list[PositionType] = [] # 読み込まれた、または新規のゴール位置を格納

        all_checkpoints_found = True # 全てのエージェントのチェックポイントが見つかったかを示すフラグ
        loaded_episode = 0 # 読み込まれたエピソード数
        loaded_epsilon = 1.0

        for agent in self.agents:
            file_path = os.path.join(checkpoint_dir, f'agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
            load_data:dict = io_handler.load(file_path) # IOHandlerを使用

            if load_data: # データが読み込まれた場合
                qtable: QTableType = load_data.get('state_dict', {}) # 存在しないキーの場合に備えてgetを使用
                agent.set_Qtable(qtable)

                # 最初の有効なチェックポイントからエピソード数とゴール位置を取得
                if not goal_pos: # goal_posがまだ設定されていない場合
                    goal_pos = load_data.get('goal_position', [])
                if loaded_episode == 0: # loaded_episodeがまだ設定されていない場合
                    loaded_episode = load_data.get('episode', 0)
                if loaded_epsilon == 0:
                    loaded_epsilon = load_data.get('epsilon', 1.0)

            else: # チェックポイントファイルが存在しない、または読み込みエラーの場合
                print(f"エージェント {agent.agent_id} のチェックポイントが見つからないか、読み込みに失敗しました。Qテーブルを初期化します。")
                #agent.reset_Qtable() # AgentクラスのQテーブル初期化メソッドを呼び出す
                agent.set_Qtable({}) # AgentクラスのQテーブル初期化メソッドを呼び出す
                all_checkpoints_found = False # 1つでも見つからなければFalse

        if all_checkpoints_found:
            self._start_episode = loaded_episode

            # εを設定
            for agent in self.agents:
                agent.epsilon = loaded_epsilon

            return goal_pos # 読み込まれたゴール位置を返す
        else:
            # 1つでもチェックポイントが見つからなかった場合、新規学習として扱う
            self._start_episode = 0
            # 新規の場合のゴール位置サンプリングは__init__で行われるため、ここでは特別な処理は不要
            return [] # 空のリストを返すことで、__init__で新規サンプリングを促す

    # def load_checkpoint_mentetyuu(self)->tuple[int, list[PositionType]]:
    #     print("チェックポイント読み込み中...")
    #     io_handler = IOHandler()
    #     checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
    #     load_data:list[dict] = []
    #     goal_pos:list[PositionType] = [] # 読み込まれた、または新規のゴール位置を格納
    #     loaded_episode:int = int(0) # 読み込まれたエピソード数

    #     for agent in self.agents:
    #         file_path = os.path.join(checkpoint_dir, f'agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
    #         data = io_handler.load(file_path)
    #         if not data:
    #             loaded_episode:int = int(0)
    #             goal_pos:list[PositionType] = []
    #             return loaded_episode, goal_pos # ないとき初期値をリターン

    #         load_data.append(data) # IOHandlerを使用

    #     for agent,data in zip(self.agents,load_data):
    #         qtable: QTableType = data.get('state_dict', {}) # 存在しないキーの場合に備えてgetを使用
    #         agent.set_Qtable(qtable)
        
    #     # エージェント0についてエピソードを取得
    #     loaded_episode:int = int(load_data[0]['episode'])
    #     goal_pos:list[PositionType] = list(load_data[0]['goal_position'])

    #     return loaded_episode, goal_pos

    def load_model(self):
        """
        ファイルから各AgentのQテーブルを読み込み、対応するAgentに設定する (推論用など).
        IOHandlerクラスを使用して読み込み処理を行う.
        """
        print("モデル読み込み中...")

        io_handler = IOHandler()
        for agent in self.agents:
            file_path = os.path.join(self.save_dir, "models", f'agent_{agent.agent_id}_model.pth') # ファイル名を変更

            qtable: QTableType = io_handler.load(file_path) # IOHandlerを使用
            if qtable: # 読み込みに成功した場合のみ設定
                agent.set_Qtable(qtable)
            else:
                print(f"エージェント {agent.agent_id} のモデルが見つからないか、読み込みに失敗しました。Qテーブルは更新されません。")

    def result_save(self):
        print("Saving results...")
        self.plot_results.draw()
        self.plot_results.draw_heatmap()
       
    def make_trajectry(self):
        states_log = []
        done = False
        time_step = 0
        reward = 0.0

        states = self.env.reset()

        for agent in self.agents: agent.epsilon = 0.0 # 無理くり探索させない

        while not done and time_step < self.max_ts:
            
            actions = []
            
            for agent in self.agents:
                a = agent.get_action(states)                #<-ここの仕様が統一感がない
                actions.append(a)
            
            next_states, r, done, _ = self.env.step(actions)#<-ここの仕様が統一感がない

            states = next_states
            states_log.append(states)

            time_step +=1
            reward += r

        return states_log, reward, done

    def render_anime(self, total_episode):
        traj, r, done = self.make_trajectry()
        print(r,done)
        render = Render(self.grid_size, self.goals_number, self.agents_number)
        render.render_anime(traj, os.path.join(self.save_dir, f"{total_episode}.gif"))

    def train(self,total_episodes:int):
        # 事前条件チェック
        if self.agents_number > self.goals_number:
            print('goals_num >= agents_num に設定してください.\n')
            sys.exit()
        
        if total_episodes <= self._start_episode:
            print(f"学習されない: {total_episodes} <= {self._start_episode}")
            return

        # 学習開始メッセージ
        # IQL/CQLの表示はAgentクラスの実装に依存するが、ここではMultiAgent_QがQ学習を orchestrate していることを示す
        # if self.mask: # Remove this if condition
        #     # Note: The mask logic might need to be handled within the Agent's learn method
        #     print(f"{GREEN}IQL/CQL (Q学習ベース) で学習中{RESET}\n")
        # else:
        print(f"{GREEN}Q学習で学習中{RESET}\n")
        print(f"ゴール位置: {self.env.get_goal_positions().values()}")

        total_step = 0

        episode_rewards: list[float] = []   # エピソードごとの報酬を格納
        episode_steps  : list[int]   = []   # エピソードごとのステップ数を格納
        episode_losses : list[float] = []   # エピソードごとの損失を格納
        done_counts    : list[int]   = []   # エピソードごとで成功/失敗を記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        print(f"---------学習開始 (全 {total_episodes} エピソード)----------")

        for episode in range(self._start_episode, total_episodes + 1):
            if episode % 100 == 0:
                print(f"Episode {episode} / {total_episodes}", end='\r', flush=True)

            # 1000エピソードごとに平均を出力
            if episode % 1000 == 0 and (episode != self._start_episode):
                print() # 改行して進捗表示をクリア
                
                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)#    if episode_losses else 0 # ここで losses は過去100エピソードの平均損失リスト
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                avg_step   = sum(episode_steps) / len(episode_steps)
                done_rate  = sum(done_counts) / len(done_counts)

                q_table_size = [agent.get_q_table_size() for agent in self.agents]

                print(f"==== エピソード {episode - 999} ~ {episode} の平均 step  : {GREEN}{avg_step:.2f}{RESET}")
                print(f"==== エピソード {episode - 999} ~ {episode} の平均 reward: {GREEN}{avg_reward:.2f}{RESET}")
                print(f"==== エピソード {episode - 999} ~ {episode} の平均 loss  : {GREEN}{avg_loss:.4f}{RESET}")
                print(f"==== エピソード {episode - 999} ~ {episode} の成功率     : {GREEN}{done_rate:.2f}{RESET}")
                print(f"==== ε: {self.agents[0].epsilon:.3f}")
                print(f"==== エピソード {episode - 999} ~ {episode} のデータ量   : {GREEN}{q_table_size}{RESET}\n")
                
                #avg_reward_temp, avg_step_temp = 0, 0
                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []
            
            if episode % 1000 == 0:
                self.save_checkpoint(episode, self.goal_pos)
                                
            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # The reset method in the updated MultiAgentGridEnv now returns observation, info
            current_states:tuple[tuple[int, int], ...] = self.env.reset()
            #print("current_states:",current_states)

            done = False
            step_count = 0
            episode_reward:float = 0.0
            step_losses:list[float] = [] # 各ステップでのエージェントごとの損失

            #self.logger.insert_episode(episode,done,episode_reward,0.0)

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions:list[int] = []
                for i, agent in enumerate(self.agents):
                    # Agentクラスのdecay_epsilon_powを呼び出し
                    agent.decay_epsilon_power(total_step)

                    # Agentクラスのget_actionを呼び出し (global_stateのみ渡す)
                    actions.append(agent.get_action(current_states))

                # エージェントの状態を保存（オプション）
                agent_positions_dict = self.env.get_agent_positions() # GridWorldのメソッドを使用
                agent_positions_list = [agent_positions_dict[agent_id] for agent_id in self.env._agent_ids]

                for i, pos in enumerate(agent_positions_list):
                    # Saver expects agent_idx, position, step
                    self.saver.log_agent_states(i, pos[0], pos[1]) # Use step_count here


                # 環境にステップを与えて状態を更新
                # The step method in the updated MultiAgentGridEnv now returns observation, reward, done, info
                next_observation, reward, done, info = self.env.step(actions)

                # オーバーヘッドが凄まじいため、いったん削除
                # if episode in [9,10,11,99,100,101,999,1000,1001,9999,10000,10001,19999,20000,20001,29999,30000,30001]:
                #     ob = self.env.get_all_object()
                #     object_data = {"name": ob.keys(), "cs": current_states, "ns": next_observation}
                #     agents_data = {"id": self.env.get_agent_positions().keys(), "action": actions, "reward": [reward]*self.agents_number}
                #     self.logger.insert_step(episode,step_count,object_data,agents_data)
                # Q学習は経験ごとに逐次更新
                # 各エージェントに対して学習を実行
                for i, agent in enumerate(self.agents):
                    loss = agent.learn(current_states, int(actions[i]), float(reward), next_observation, bool(done))
                    step_losses.append(loss)

                current_states = next_observation # 状態を更新
                episode_reward += reward
                step_count += 1

            # エピソード終了
            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses)
            episode_step:int = step_count

            # ログにスコアを記録
            self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, done)
            #self.logger.end_episode(episode,done,episode_reward,episode_loss)

            episode_losses.append(episode_loss) # 100エピソードまで貯め続ける
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(done)
            total_step += episode_step

        #self.saver.save_remaining_episode_data() <- 2000-2000が生成される原因になる。この半端なエピドートは捨てることで解決を図る。
        self.saver.save_visited_coordinates()
        # self.logger.flush_buffer()
        self.save_model()
        print(f"---------学習終了 (全 {total_episodes} エピソード完了)----------")

    # def log_disp(self):
    #     episode_data = self.logger.get_episode_data()
    #     episode_data = episode_data[::5]
    #     # detail_show_list = [1,2,3,4,5,180,181,182,195,196,197,198,199,200] # 詳細を見るエピソードを指定
    #     for data in episode_data:
    #         print(f"{GREEN}Episode: {data[0]:-4}, done: {bool(data[1])}, reward: {data[2]:.4}, loss: {data[3]:.4}{RESET}")
