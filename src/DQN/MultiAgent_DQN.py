import sys
import os

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
        # 3つまでゴールを手動で設定できるように変更。
        fix_goal_pool = [(args.grid_size-1,args.grid_size-1),(args.grid_size//4,args.grid_size//3)]
        self._fix_goal_from_goal_number = fix_goal_pool[:min(args.goals_number, len(fix_goal_pool))]

        self.env = MultiAgentGridEnv(args, fixrd_goals=self._fix_goal_from_goal_number)

        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_number = args.agents_number
        self.goals_number = args.goals_number # Fixed: Changed args.goals_num to args.goals_number
        self.grid_size = args.grid_size
        # self.load_model = args.load_model
        self.mask = args.mask
        self.update_frequency = 4# 学習の頻度

        self.save_agent_states = args.save_agent_states

        self.start_episode = 1

        # 結果保存ディレクトリの設定と作成
        folder_name = "DQN_"
        if args.mask==1 or args.neighbor_distance==0:
            # mask==1: IQL
            folder_name+="IQL"
        else:
            # mask==0: CQL
            folder_name+="IQL"
            if args.neighbor_distance < self.grid_size:
                folder_name += "観測"
                folder_name += f"[{args.neighbor_distance}]"
            else:
                folder_name += "全観測"

        folder_name += f"_報酬[{self.reward_mode}]_[{self.grid_size}x{self.grid_size}]_T[{self.max_ts}]_A-G[{self.agents_number}-{self.goals_number}]"
        self.save_dir = os.path.join(
            "output",
            # f"DQN_mask[{args.mask}]_Reward[{args.reward_mode}]_env[{args.grid_size}x{args.grid_size}]_max_ts[{args.max_timestep}]_agents[{args.agents_number}]" + (f"_PER_alpha[{args.alpha}]_beta_anneal[{args.beta_anneal_steps}]" if args.use_per else "")
            folder_name
        )

        # 結果保存およびプロット関連クラスの初期化
        self.saver = Saver(self.save_dir,self.grid_size)
        self.saver.CALCULATION_PERIOD = 50
        self.plot_results = PlotResults(self.save_dir)

        self.load_checkpoint(None)

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
        episode_rewards: list[float] = []   # エピソードごとの報酬を格納
        episode_steps  : list[int]   = []   # エピソードごとのステップ数を格納
        episode_losses : list[float] = []   # エピソードごとの損失を格納
        done_counts    : list[int]   = []   # エピソードごとで成功/失敗を記録

        # ----------------------------------
        # メインループ（各エピソード）
        # ----------------------------------
        for episode in range(self.start_episode, self.episode_num + 1):

            print('■', end='',flush=True)  # 進捗表示 (エピソード100回ごとに改行)

            # 50エピソードごとに集計結果を出力
            CONSOLE_LOG_FREQ = 50
            if (episode % 50 == 0) and (episode!=self.start_episode):
                print() # 改行して進捗表示をクリア

                # エピソードごとの平均損失、平均ステップ、平均報酬を計算し、表示に追加
                avg_loss   = sum(episode_losses) / len(episode_losses)      # 期間内の平均損失
                avg_reward = sum(episode_rewards) / len(episode_rewards)    # 期間内の平均報酬
                avg_step   = sum(episode_steps) / len(episode_steps)        # 期間内の平均ステップ数
                done_rate  = sum(done_counts) / len(done_counts)            # 達成率

                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 step  : {GREEN}{avg_step:.3f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 reward: {GREEN}{avg_reward:.3f}{RESET}")
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の達成率     : {GREEN}{done_rate:.2f}{RESET}") # 達成率も出力 .2f で小数点以下2桁表示
                print(f"     エピソード {episode - CONSOLE_LOG_FREQ+1} ~ {episode} の平均 loss  : {GREEN}{avg_loss:.5f}{RESET}") # 平均損失も出力
                if self.agents[0].model.use_per:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.agents[0].epsilon:.3f}{RESET}, beta: {GREEN}{self.agents[0].beta:.3f}{RESET}")
                else:
                    print(f"     (Step: {total_step}), 探索率 : {GREEN}{self.agents[0].epsilon:.3f}{RESET}")

                episode_losses = [] # 100エピソードごとに損失リストもリセット
                episode_rewards = []
                episode_steps  = []
                done_counts = []

            if episode % 50 == 0:
                self.save_checkpoint(episode)

            # 各エピソード開始時に環境をリセット
            iap = [(0,i) for i in range(self.agents_number)]
            current_global_state = self.env.reset(initial_agent_positions=iap)

            done = False # エピソード完了フラグ
            step_count:int = 0 # 現在のエピソードのステップ数
            episode_reward:float = 0.0 # 現在のエピソードの累積報酬

            step_losses:list[float] = []

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
                episode_reward += reward

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
                            step_losses.append(current_loss)

                # 全体状態を次の状態に更新
                current_global_state = next_global_state

                step_count += 1 # エピソード内のステップ数をインクリメント
                total_step += 1 # 全体のステップ数をインクリメント

            # ---------------------------
            # エピソード終了後の処理
            # ---------------------------

            # エピソードが完了 (done == True) した場合、達成エピソード数カウンタをインクリメント
            if done:
                episode_steps.append(step_count) # 達成した場合のステップ数のみ加算

            # エピソードの平均損失を計算
            episode_loss:float = sum(step_losses)/len(step_losses) if step_losses else 0.0
            episode_step:int = step_count

            # Saverでエピソードごとのスコアをログに記録
            # エピソード番号、最終ステップ数、累積報酬、エピソード中の平均損失を記録
            self.saver.log_episode_data(episode, step_count, episode_reward, episode_loss, done)

            # 集計期間内の平均計算のための累積 (avg_reward_temp accumulation)
            episode_losses.append(episode_loss) # 100エピソードまで貯め続ける
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            done_counts.append(done)

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
            # agent.model.qnet.load_state_dict(loaded_state_dict)
            # agent.model.qnet_target.load_state_dict(loaded_state_dict)
            agent.set_weights_for_inference(loaded_state_dict)

    def simulate_agent_behavior(self, num_simulation_episodes: int = 1, max_simulation_timestep:int =-1):
        """
        学習済みモデルを使ってエージェントの行動をシミュレーションします。

        Args:
            num_simulation_episodes (int): シミュレーションを実行するエピソード数。
            max_simulation_timestep (Optional[int]): 各シミュレーションエピソードの最大ステップ数。
                                                     Noneの場合、MultiAgent_DQNのmax_tsが使用されます。
        """
        print(f"{GREEN}--- シミュレーション開始 (学習済みモデル使用) ---" + f"{RESET}")
        print(f"シミュレーションエピソード数: {num_simulation_episodes}\n")

        if max_simulation_timestep ==-1:
            max_simulation_timestep = self.max_ts

        self.load_model_weights()

        for episode_idx in range(1, num_simulation_episodes + 1):
            print(f"--- シミュレーションエピソード {episode_idx} / {num_simulation_episodes} ---")

            # エージェントの初期位置を設定
            current_global_state = self.env.reset(initial_agent_positions=[(0,i) for i in range(self.agents_number)]) 
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

    def load_checkpoint(self, episode:int|None=None):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "checkpoints")
        
        if not os.path.exists(model_dir):
            print(f"{model_dir}は存在しなかったため、新規学習とする")
            return

        try:
            for agent in self.agents:            
                if episode:
                    file_path = os.path.join(model_dir, f"checkpoint_episode[{episode}]_id[{agent.agent_id}].pth")
                else:
                    file_path = os.path.join(model_dir, f"checkpoint_{agent.agent_id}.pth")

                # Model_IOから辞書データを取得
                # ※Model_IO側でtarget_stateも返すように修正が必要（後述）

                q_dict, t_dict, optim_dict, epoch, epsilon = model_io.load_checkpoint(file_path)
                
                # エージェントの状態を一括更新
                agent.set_weights_for_training(q_dict, t_dict, optim_dict, epsilon)
                self.start_episode = epoch +1 # Trainerのエピソード数も同期
        except:
            Exception("load_checkpointメソッドでのエラー")
        
        print(f"エピソード{self.start_episode}から再開")

    def save_checkpoint(self, episode:int):
        model_io = Model_IO()
        model_dir = os.path.join(self.save_dir, "checkpoints")
        if not os.path.exists(model_dir): os.makedirs(model_dir)

        for agent in self.agents:
            # DQNModelから各state_dictを取得
            q_dict, t_dict, optim_dict = agent.get_weights()
            file_path = os.path.join(model_dir, f"checkpoint_{agent.agent_id}.pth")
            file_path_episode = os.path.join(model_dir, f"checkpoint_episode[{episode}]_id[{agent.agent_id}].pth")
            
            # 最新と現在のエピソードの2ファイルを書き込み
            model_io.save_checkpoint(file_path, q_dict, t_dict, optim_dict, episode, agent.epsilon)
            model_io.save_checkpoint(file_path_episode, q_dict, t_dict, optim_dict, episode, agent.epsilon)
