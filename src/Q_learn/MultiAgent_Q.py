import sys
import os

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


from Enviroments.MultiAgentGridEnv import MultiAgentGridEnv
from utils.Saver import Saver
from utils.plot_results import PlotResults
#from utils.ConfigManager import ConfigManager, ConfigLoader
from utils.IO_Handler import IOHandler
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
            f"{Q_Strategy}_r[{self.reward_mode}]_env[{self.grid_size}x{self.grid_size}]_max_ts[{self.max_ts}]_agents[{self.agents_number}]_goals[{self.goals_number}]"
        )

        self.io_handler = IOHandler() # IOHandlerのインスタンスを作成

        self._start_episode = 0

        #goal_pos_list:list[PositionType] = self.load_checkpoint()
        self._start_episode, goal_pos_list = self.load_checkpoint_mentetyuu()
        self.env = MultiAgentGridEnv(args, goal_pos_list)# 環境クラス
        
        self.goal_pos = tuple(self.env.get_goal_positions().values())
        
        print("pos", self.goal_pos)
        print("Qtable-len: ", self.agents[0].get_q_table_size())

        # 学習で発生したデータを保存するクラス
        self.saver = Saver(self.save_dir,self.grid_size)
        
        # saverクラスで保存されたデータを使ってグラフを作るクラス
        self.plot_results = PlotResults(self.save_dir)


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
        print(f"ゴール位置: {self.env.get_goal_positions().values()}")

        total_step = 0
        #avg_reward_temp, avg_step_temp = 0, 0
        episode_rewards: list[float] = []   # エピソードごとの報酬を格納
        episode_steps  : list[int]   = []   # エピソードごとのステップ数を格納
        episode_losses : list[float] = []   # エピソードごとの損失を格納
        done_counts    : list[int]   = []   # エピソードごとで成功/失敗を記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode_num in range(1, self.episode_number + 1):
            if episode_num % 10 == 0:
                print(f"Episode {episode_num} / {self.episode_number}", end='\r', flush=True)

            # 100エピソードごとに平均を出力
            if episode_num % 100 == 0:
                print() # 改行して進捗表示をクリア
                #avg_reward = avg_reward_temp / 100
                #avg_step = avg_step_temp / 100
                
                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)#    if episode_losses else 0 # ここで losses は過去100エピソードの平均損失リスト
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                avg_step   = sum(episode_steps) / len(episode_steps)
                done_rate  = sum(done_counts) / len(done_counts)

                q_table_size = [agent.get_q_table_size() for agent in self.agents]

                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 loss  : {GREEN}{avg_loss:.4f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の成功率     : {GREEN}{done_rate:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} のデータ量   : {GREEN}{q_table_size}{RESET}\n")
                
                #avg_reward_temp, avg_step_temp = 0, 0
                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []
                
            # --------------------------------------------
            # 各エピソード開始時に環境をリセット
            # これによりエージェントが再配置される
            # --------------------------------------------
            # The reset method in the updated MultiAgentGridEnv now returns observation, info
            current_states:tuple[tuple[int, int], ...] = self.env.reset()

            done = False
            step_count = 0
            #episode_steps  = []
            #episode_losses = [] # エピソードごとの損失を格納
            episode_reward:float = 0.0
            step_losses:list[float] = [] # 各ステップでのエージェントごとの損失

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions:list[int] = []
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
                #step_losses:list[float] = [] # 各ステップでのエージェントごとの損失

                # 各エージェントに対して学習を実行
                for i, agent in enumerate(self.agents):
                    #agent_action = debug_actions[i]
                    loss = agent.learn(current_states, int(actions[i]), float(reward), next_observation, bool(done))
                    step_losses.append(loss)

                current_states = next_observation # 状態を更新
                #step_reward += reward
                episode_reward += reward
                step_count += 1
                total_step += 1

            # エピソード終了
            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses)
            episode_step:int = step_count

            # ログにスコアを記録
            self.saver.log_episode_data(episode_num, step_count, episode_reward, episode_loss, done)

            episode_losses.append(episode_loss) # 100エピソードまで貯め続ける
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(done)

            #avg_reward_temp += episode_reward
            #avg_step_temp += step_count

        self.saver.save_remaining_episode_data()
        self.saver.save_visited_coordinates()
        print()  # 終了時に改行


    def save_checkpoint(self, episode:int, goal_position:tuple[PositionType,...]|list[PositionType]):
        """
        保持している各AgentのQテーブルと学習状態をチェックポイントファイルに保存する.
        IOHandlerクラスを使用して保存処理を行う.
        """
        print("チェックポイント保存中...")
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        temp_path_list = []
        for agent in self.agents:
            q_table_data = agent.get_Qtable()
            temp_path = os.path.join(checkpoint_dir, f'temp_agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
            temp_path_list.append(temp_path)
            save_param = {"state_dict":q_table_data, "episode": episode, "goal_position": goal_position}
            self.io_handler.save(save_param, temp_path) # IOHandlerを使用

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
        model_dir = os.path.join(self.save_dir, "models") # ファイルパス生成方法を修正
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for agent in self.agents:
            q_table_data = agent.get_Qtable()
            file_path = os.path.join(model_dir, f'agent_{agent.agent_id}_model.pth') # ファイル名を変更
            self.io_handler.save(q_table_data, file_path) # IOHandlerを使用

    def load_checkpoint(self)->list[PositionType]:
        """
        ファイルから各AgentのQテーブルと学習状態を読み込み、対応するAgentに設定する.
        IOHandlerクラスを使用して読み込み処理を行う.
        """
        print("チェックポイント読み込み中...")
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        goal_pos:list[PositionType] = [] # 読み込まれた、または新規のゴール位置を格納

        all_checkpoints_found = True # 全てのエージェントのチェックポイントが見つかったかを示すフラグ
        loaded_episode = 0 # 読み込まれたエピソード数

        for agent in self.agents:
            file_path = os.path.join(checkpoint_dir, f'agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
            load_data:dict = self.io_handler.load(file_path) # IOHandlerを使用

            if load_data: # データが読み込まれた場合
                qtable: QTableType = load_data.get('state_dict', {}) # 存在しないキーの場合に備えてgetを使用
                agent.set_Qtable(qtable)

                # 最初の有効なチェックポイントからエピソード数とゴール位置を取得
                if not goal_pos: # goal_posがまだ設定されていない場合
                    goal_pos = load_data.get('goal_position', [])
                if loaded_episode == 0: # loaded_episodeがまだ設定されていない場合
                    loaded_episode = load_data.get('episode', 0)

            else: # チェックポイントファイルが存在しない、または読み込みエラーの場合
                print(f"エージェント {agent.agent_id} のチェックポイントが見つからないか、読み込みに失敗しました。Qテーブルを初期化します。")
                #agent.reset_Qtable() # AgentクラスのQテーブル初期化メソッドを呼び出す
                agent.set_Qtable({}) # AgentクラスのQテーブル初期化メソッドを呼び出す
                all_checkpoints_found = False # 1つでも見つからなければFalse

        if all_checkpoints_found:
            self._start_episode = loaded_episode
            return goal_pos # 読み込まれたゴール位置を返す
        else:
            # 1つでもチェックポイントが見つからなかった場合、新規学習として扱う
            self._start_episode = 0
            # 新規の場合のゴール位置サンプリングは__init__で行われるため、ここでは特別な処理は不要
            return [] # 空のリストを返すことで、__init__で新規サンプリングを促す

    def load_checkpoint_mentetyuu(self)->tuple[int, list[PositionType]]:
        print("チェックポイント読み込み中...")
        checkpoint_dir = os.path.join(self.save_dir, ".checkpoints")
        load_data:list[dict] = []
        goal_pos:list[PositionType] = [] # 読み込まれた、または新規のゴール位置を格納
        loaded_episode:int = int(0) # 読み込まれたエピソード数

        for agent in self.agents:
            file_path = os.path.join(checkpoint_dir, f'agent_{agent.agent_id}_checkpoint.pth') # ファイル名を変更
            data = self.io_handler.load(file_path)
            if not data:
                loaded_episode:int = int(0)
                goal_pos:list[PositionType] = []
                return loaded_episode, goal_pos # ないとき初期値をリターン

            load_data.append(data) # IOHandlerを使用

        for agent,data in zip(self.agents,load_data):
            qtable: QTableType = data.get('state_dict', {}) # 存在しないキーの場合に備えてgetを使用
            agent.set_Qtable(qtable)
        
        # エージェント0についてエピソードを取得
        loaded_episode:int = int(load_data[0]['episode'])
        goal_pos:list[PositionType] = list(load_data[0]['goal_position'])

        return loaded_episode, goal_pos

    def load_model(self):
        """
        ファイルから各AgentのQテーブルを読み込み、対応するAgentに設定する (推論用など).
        IOHandlerクラスを使用して読み込み処理を行う.
        """
        print("モデル読み込み中...")

        for agent in self.agents:
            file_path = os.path.join(self.save_dir, "models", f'agent_{agent.agent_id}_model.pth') # ファイル名を変更

            qtable: QTableType = self.io_handler.load(file_path) # IOHandlerを使用
            if qtable: # 読み込みに成功した場合のみ設定
                agent.set_Qtable(qtable)
            else:
                print(f"エージェント {agent.agent_id} のモデルが見つからないか、読み込みに失敗しました。Qテーブルは更新されません。")

    def result_save(self):
        print("Saving results...")
        self.plot_results.draw()
        self.plot_results.draw_heatmap()
       

    def debug_train(self):
        class DebugGridWorldConfig:
            def __init__(self):
                self.grid_size = 5  # 小さなグリッド
                # デバッグしたい特定のエピソード番号のリスト
                self.debug_episodes = [110, 7000]
                # 最大エピソード数はデバッグしたい最大のエピソード番号までとします
                self.episode_number = max(self.debug_episodes)
                self.max_timestep = 25 # ステップ数は適宜調整
                self.mask = 0 # 自己中心モード (状態空間が小さい方がデバッグしやすい)
                # エージェント数を2に戻して複数のエージェントのTD誤差を確認できるようにします
                self.agents_number = 2
                self.goals_number = 2 # ゴール2つ
                self.reward_mode = 0 # シンプルな報酬モード
                self.render_mode = 0
                self.load_model = 0
                self.pause_duration = 0.1
                self.learning_rate = 0.1
                self.discount_factor = 0.99
                self.epsilon = 0.1 # 探索率を低めに設定（再現性を高めるため）
                self.epsilon_decay_alpha = 0.70
                self.min_epsilon = 0.01
                self.max_epsilon = 1.0 # εを高くして探索を促す
                self.action_size = 5

        debug_config = DebugGridWorldConfig()

        # ダミーのエージェントをインスタンス化
        # debug_config.agents_number の数だけ Agent インスタンスを作成します
        debug_agents = [Agent(debug_config, i) for i in range(debug_config.agents_number)]

        # QTableを初期化 (ロードはしない)
        # 各エージェントの Agent.__init__ で QTable は初期化されています
        # ここでは特に明示的な初期化コードは不要です

        # ダミーの環境をインスタンス化
        debug_env = MultiAgentGridEnv(debug_config, tuple(self.env.get_goal_positions().values()))

        episode_losses:list[float] = []#100エピソード分のTDを格納
        episode_reward = 0

        print("--- TD誤差 デバッグ開始 ---")

        # 各エピソードのループ
        for episode_num in range(1, debug_config.episode_number + 1):

            # デバッグ対象のエピソードかどうかを判定
            is_debug_episode = episode_num in debug_config.debug_episodes

            if is_debug_episode:
                print(f"\n--- エピソード {episode_num} (デバッグ対象) ---")
            elif episode_num % 100 == 0:
                average_episode100_loss = sum(episode_losses)/len(episode_losses)
                print(f" エピソード {episode_num} / {debug_config.episode_number}", end='\r', flush=True)
                print(f" エピソード {episode_num - 99} ~ {episode_num} の平均 loss  : {GREEN}{(average_episode100_loss):.4f}{RESET}")
                print(f" エピソード {episode_num - 99} ~ {episode_num} のloss/25step: {(episode_losses[::25])}")

                episode_losses = []


            # 環境をリセットし、初期状態を取得
            try:
                # エージェントをゴールと重複しないように、ランダムに配置
                debug_current_states:tuple[tuple[int, int], ...] = debug_env.reset() # ランダム配置を使用
                if is_debug_episode:
                    print(f"初期状態 (グローバル): {debug_current_states}")
            except ValueError as e:
                print(f"環境リセットエラー (エピソード {episode_num}): {e}. ゴールとエージェントの数がグリッドサイズに対して多すぎる可能性があります。設定を確認してください。")
                continue # 次のエピソードへ


            done = False
            step_count = 0

            # 1エピソードのステップループ
            while not done and step_count < debug_config.max_timestep:
                # 各エージェントのアクションを選択し、リストに収集
                debug_actions = []
                for i, agent in enumerate(debug_agents):
                    # エージェントのε減衰 (全体のステップ数を使用)
                    # あるいは、デバッグでは固定εを使用しても良い
                    # total_step の計算はメインループに依存するので、ここではエピソード内ステップを使用するか固定します
                    # agent.decay_epsilon_pow(episode_num * debug_config.max_timestep + step_count) # シンプルなステップ数を使う場合
                    agent.epsilon = 0 if is_debug_episode else 0.1

                    # 現在のQテーブル用の状態表現を取得 (Agent._get_q_stateはAgentインスタンスのメソッド)
                    # global_state を渡して各エージェントが自身のQStateを生成
                    agent_q_state = agent._get_q_state(debug_current_states)

                    # 行動を選択 (epsilon-greedy)
                    # Agent.get_action は global_state を受け取り、内部でQStateに変換して行動を選択
                    debug_actions.append(agent.get_action(debug_current_states))


                # 環境を1ステップ進める (全エージェントのアクションをリストで渡す)
                debug_next_states, debug_reward, debug_done, debug_info = debug_env.step(debug_actions)

                step_losses:list[float] = [] # 各ステップでのエージェントごとの損失
                # --- TD誤差の計算とログ出力 (デバッグ対象エピソードのみ) ---
                if is_debug_episode:
                    print(f"\n-- ステップ {step_count + 1} --")
                    print(f"現在のグローバル状態: {debug_current_states}")
                    print(f"選択された行動 (全エージェント): {debug_actions}")
                    print(f"得られた報酬: {debug_reward}")
                    print(f"次のグローバル状態: {debug_next_states}")
                    print(f"エピソード完了フラグ (done): {debug_done}")

                    # 各エージェントについてTD誤差を計算し、ログ出力
                    for i, agent in enumerate(debug_agents):
                        agent_q_state = agent._get_q_state(debug_current_states)
                        agent_next_q_state = agent._get_q_state(debug_next_states)
                        agent_action = debug_actions[i] # このエージェントが取った行動

                        print(f"\n  -- エージェント {i} のTD誤差計算 --")
                        print(f"  現在のQテーブル用状態: {agent_q_state}")

                        # Qテーブル内部の状態を確認
                        print(f"  現在の状態 ({agent_q_state}) はQテーブルに存在するか: {agent_q_state in agent.q_table.q_table}")
                        if agent_q_state in agent.q_table.q_table:
                            print(f"  Qテーブル内部の現在の状態のQ値: {agent.q_table.q_table[agent_q_state]}")
                        print(f"  Agent.q_table.get_q_values({agent_q_state}) が返すQ値: {agent.q_table.get_q_values(agent_q_state)}")


                        print(f"  選択された行動: {agent_action}")
                        # 各エージェントが受け取る報酬はグローバル報酬と仮定（MultiAgent_Q参照）
                        print(f"  受け取った報酬: {debug_reward}")
                        print(f"  次のQテーブル用状態: {agent_next_q_state}")

                        # Qテーブル内部の次の状態を確認
                        print(f"  次の状態 ({agent_next_q_state}) はQテーブルに存在するか: {agent_next_q_state in agent.q_table.q_table}")
                        if agent_next_q_state in agent.q_table.q_table:
                            print(f"  Qテーブル内部の次の状態のQ値: {agent.q_table.q_table[agent_next_q_state]}")


                        # 現在の状態・行動に対するQ値 (更新前の値)
                        # get_q_values を使用して、Qテーブルにまだ存在しない状態がアクセスされてもエラーにならないようにします
                        debug_current_q_value = agent.q_table.get_q_values(agent_q_state)[agent_action]
                        print(f"  現在のQ(s, a) (更新前): {debug_current_q_value}")

                        # 次の状態での最大Q値 (Q学習)
                        # エピソードが完了した場合は次の状態の価値は0
                        debug_max_next_q_value = 0.0
                        if not debug_done:
                            debug_max_next_q_value = agent.q_table.get_max_q_value(agent_next_q_state) # QTable.get_max_q_valueを使用
                        print(f"  次の状態での最大Q(s', a'): {debug_max_next_q_value}")


                        # TDターゲットの計算
                        debug_td_target = debug_reward + debug_config.discount_factor * debug_max_next_q_value
                        print(f"  TDターゲット (r + γ * max Q(s', a')): {debug_td_target:.3}")

                        # TDデルタ (誤差) の計算
                        debug_td_delta = debug_td_target - debug_current_q_value
                        print(f"  TDデルタ (TDターゲット - Q(s, a)): {debug_td_delta:.3}")

                        # TD誤差絶対値 (Lossとして使用される値)
                        debug_loss = abs(debug_td_delta)
                        print(f"  TD誤差絶対値 (Loss): {debug_loss:.3}")

                        # Q値の更新 (Agent.learnで行われる処理) の直前と直後の値を確認
                        # 更新前の該当Q値を取得 (再度取得して厳密に)
                        q_before_learn = agent.q_table.get_q_values(agent_q_state)[agent_action]
                        print(f"  Q(s, a) 更新直前: {q_before_learn:.4}")

                        # 学習実行
                        # Agent.learn は QTable.learn をラップしているため、Agent.learn を呼び出します
                        actual_td_delta_from_learn = agent.learn(debug_current_states, agent_action, debug_reward, debug_next_states, debug_done)

                        # 更新後の該当Q値を取得
                        q_after_learn = agent.q_table.get_q_values(agent_q_state)[agent_action]
                        print(f"  Q(s, a) 更新直後: {q_after_learn:.4}")
                        # 期待される更新後の値も計算して比較するとさらに良い
                        #expected_q_after_learn = q_before_learn + debug_config.learning_rate * debug_td_delta
                        #print(f"  期待される更新後のQ(s, a): {expected_q_after_learn}")
                        #print(f"  期待値と実際の更新値の差: {abs(q_after_learn - expected_q_after_learn)}")
                        print(f"  Agent.learn から返されたTD誤差絶対値: {abs(actual_td_delta_from_learn)}")
                        step_losses.append(actual_td_delta_from_learn)

                else: # デバッグ対象外エピソードでも学習は実行
                    # 各エージェントに対して学習を実行
                    for i, agent in enumerate(debug_agents):
                        agent_action = debug_actions[i]
                        loss = agent.learn(debug_current_states, agent_action, debug_reward, debug_next_states, debug_done)
                        step_losses.append(loss)
                
                if episode_num%100==0:print(episode_num,"step_losses:",step_losses)

                # 次のステップのために状態を更新
                debug_current_states = debug_next_states

                step_count += 1
                episode_reward += debug_reward
                # エピソードが完了したらループを抜ける
                if debug_done:
                    if is_debug_episode:
                        print("\nエピソード完了.")
                    break
            
            # エピソード終了
            episode_loss:float = (sum(step_losses)/len(step_losses))
            episode_losses.append(episode_loss)
            # ログにスコアを記録
            self.saver.log_episode_data(episode_num, step_count, episode_reward, episode_loss, done)


        # デバッグ後のQテーブルサイズ確認 (任意)
        print(f"\n--- TD誤差 デバッグ終了 ---")
        # 各エージェントのQテーブルサイズを出力
        for i, agent in enumerate(debug_agents):
            print(f"エージェント {i} の最終エピソード後のQテーブルサイズ: {agent.get_q_table_size()}")


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

        #total_step = 0
        #avg_reward_temp, avg_step_temp = 0, 0
        episode_rewards: list[float] = []   # エピソードごとの報酬を格納
        episode_steps  : list[int]   = []   # エピソードごとのステップ数を格納
        episode_losses : list[float] = []   # エピソードごとの損失を格納
        done_counts    : list[int]   = []   # エピソードごとで成功/失敗を記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        print(f"---------学習開始 (全 {total_episodes} エピソード)----------")

        for episode_num in range(self._start_episode, total_episodes + 1):
            if episode_num % 100 == 0:
                print(f"Episode {episode_num} / {total_episodes}", end='\r', flush=True)

            # 1000エピソードごとに平均を出力
            if episode_num % 1000 == 0 and (episode_num != self._start_episode):
                print() # 改行して進捗表示をクリア
                
                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)#    if episode_losses else 0 # ここで losses は過去100エピソードの平均損失リスト
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                avg_step   = sum(episode_steps) / len(episode_steps)
                done_rate  = sum(done_counts) / len(done_counts)

                q_table_size = [agent.get_q_table_size() for agent in self.agents]

                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 step  : {GREEN}{avg_step:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 reward: {GREEN}{avg_reward:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の平均 loss  : {GREEN}{avg_loss:.4f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} の成功率     : {GREEN}{done_rate:.2f}{RESET}")
                print(f"==== エピソード {episode_num - 99} ~ {episode_num} のデータ量   : {GREEN}{q_table_size}{RESET}\n")
                
                #avg_reward_temp, avg_step_temp = 0, 0
                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []
            
            if episode_num % 1000 == 0:
                self.save_checkpoint(episode_num, self.goal_pos)
                                
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

            # ---------------------------
            # 1エピソードのステップループ
            # ---------------------------
            while not done and step_count < self.max_ts:
                actions:list[int] = []
                for i, agent in enumerate(self.agents):
                    # Agentクラスのdecay_epsilon_powを呼び出し
                    agent.decay_epsilon_pow(episode_num)

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

                # Q学習は経験ごとに逐次更新
                # 各エージェントに対して学習を実行
                for i, agent in enumerate(self.agents):
                    loss = agent.learn(current_states, int(actions[i]), float(reward), next_observation, bool(done))
                    step_losses.append(loss)

                current_states = next_observation # 状態を更新
                episode_reward += reward
                step_count += 1
                #total_step += 1

            # エピソード終了
            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses)
            episode_step:int = step_count

            # ログにスコアを記録
            self.saver.log_episode_data(episode_num, step_count, episode_reward, episode_loss, done)

            episode_losses.append(episode_loss) # 100エピソードまで貯め続ける
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(done)

        #self.saver.save_remaining_episode_data() <- 2000-2000が生成される原因になる。この半端なエピドートは捨てることで解決を図る。
        self.saver.save_visited_coordinates()
        self.save_model()
        print(f"---------学習終了 (全 {total_episodes} エピソード完了)----------")

    def run_method(self, total_episodes):
        print(f"---------学習開始 (全 {total_episodes} エピソード)----------")
        for episode in range(self._start_episode, total_episodes):
            # TODO: ここに各エピソードの開始処理を追加する (環境のリセットなど)

            # TODO: ここに各ステップの処理を追加する
            # - エージェントの行動選択
            # - 環境との相互作用 (ステップ実行)
            # - 報酬と次の状態の取得
            # - Qテーブルの更新

            # 例: 各ステップの処理（ダミー）
            # current_state = env.reset()
            # done = False
            # while not done:
            #     action = agent.choose_action(current_state, ...)
            #     next_state, reward, done, _ = env.step(action)
            #     agent.update_q_table(current_state, action, reward, next_state, ...)
            #     current_state = next_state

            # 定期的にチェックポイントを保存 (例: 100エピソードごと)
            checkpoint_interval = 500
            if (episode + 1) % checkpoint_interval == 0:
                print(f"--- チェックポイント保存中 (エピソード {episode + 1}) ---")
                self.save_checkpoint(episode + 1, tuple(self.env.get_goal_positions().values()))

            # TODO: 各エピソードの終了処理を追加する (ログ記録など)

        # 学習終了時に最終モデルを保存
        print(f"---------学習終了 (全 {total_episodes} エピソード完了)----------")
        self.save_model()