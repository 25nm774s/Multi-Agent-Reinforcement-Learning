import numpy as np
import os
import csv
import pandas as pd

class Saver:
    """
    学習中に発生する様々なデータをファイルに保存するためのクラス.

    エピソードごとのスコア、エージェントの訪問座標、モデルの重みなどを管理し、
    指定されたディレクトリに保存します。
    エピソードスコアは100エピソードごとに集計して保存する機能も持ちます。
    """

    # 集計期間
    CALCULATION_PERIOD:int = 100

    def __init__(self, score_summary_path, visited_coordinates_path, grid_size, position_validator_func=None):
        """
        Saverクラスの新しいインスタンスを初期化します。

        Args:
            save_dir (str): データを保存するディレクトリのパス。
            grid_size (int): エージェントの訪問座標を記録するグリッドのサイズ (NxN)。
            position_validator_func (callable, optional): 座標の有効性をチェックする関数。
                                                           引数として(x, y)のタプルを受け取り、boolを返します。
                                                           指定しない場合は、内部でGridクラスを使用します。
        """
        self.grid_size = grid_size # グリッドサイズを保存
        
        # 集計エピソード指標を保存するパス
        # self.episode_batch_summary_path = os.path.join(self.save_dir, f"aggregated_episode_metrics_{self.CALCULATION_PERIOD}.csv")
        self.episode_batch_summary_path = score_summary_path

        # ヒートマップデータ用にファイル名と形式を.npyに変更
        # self.visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")
        self.visited_coordinates_path = visited_coordinates_path

        if not os.path.exists(self.episode_batch_summary_path):
            with open(self.episode_batch_summary_path, "w", newline='') as f:
                csv.writer(f).writerow(["episode_group_start", "episode_group_end", "avg_time_step_100", "avg_reward_100", "avg_loss_100", "done_rate"])

        self.visited_count_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        self.episode_data_buffer = []
        self.episode_data_counter = 0

        self.visited_updates_counter = 0

        # 依存性注入ポイント: position_validator_funcが指定されればそれを使う
        # なければデフォルトでGridクラスを利用する
        if position_validator_func is None:
            # Gridクラスに頼らず、内部でラムダ関数を定義
            # 座標の有効性をチェックするラムダ関数
            self.is_position_valid = lambda pos: 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        else:
            self.is_position_valid = position_validator_func


    def save_q_table(self, agents, mask):
        """
        Qテーブルを保存します (現在ダミー実装).

        Args:
            agents: Qテーブルを持つエージェントのリストまたは辞書。
            mask: 保存マスク (任意)。
        """
        print('Qテーブル保存プロセスはModel_IOに移行')

    def save_dqn_weights(self, agents):
        """
        DQNモデルの重みを保存します (現在ダミー実装).

        Args:
            agents: DQNモデルを持つエージェントのリストまたは辞書。
        """
        print('モデル重み保存プロセスはModel_IOに移行')

    # グリッド内のカウントをインクリメントするように修正
    def log_agent_states(self, agent_id, x, y):
        """
        エージェントの現在の座標を記録し、訪問回数グリッドを更新します。

        座標がグリッドの範囲外の場合は警告が表示されます。

        Args:
            agent_id (int): エージェントのID。
            x (int): エージェントの現在のX座標。
            y (int): エージェントの現在のY座標。
        """
        # 注入された検証関数を使用
        if not self.is_position_valid((x,y)):
            raise ValueError(f'エージェント状態座標が無効です: x={x}, y={y} (エージェントID: {agent_id})')

        self.visited_count_grid[y, x] += 1
        self.visited_updates_counter += 1


    # 個々のエピソードデータを記録し、平均のログ記録をトリガーする新しいメソッド
    def log_episode_data(self, episode: int, time_step: int, reward: float, loss: float, done: bool):
        """
        個々のエピソードデータをバッファに記録し、100エピソードごとに集計してファイルに保存します。

        Args:
            episode (int): エピソード番号。
            time_step (int): そのエピソードでのステップ数。
            reward (float): そのエピソードで得られた報酬の合計。
            loss (float): そのエピソードでの学習損失 (該当する場合)。
            done (bool): そのエピソードを終了したか。
        """
        # 個々のエピソードデータをバッファに追加
        self.episode_data_buffer.append({'episode': episode, 'time_step': time_step, 'reward': reward, 'loss': loss, 'done':int(done)})
        # カウンターをインクリメント
        self.episode_data_counter += 1

        # カウンターが100の倍数で、かつ0より大きいかチェック
        if self.episode_data_counter % self.CALCULATION_PERIOD == 0 and self.episode_data_counter > 0:
            # バッファをDataFrameに変換
            buffer_df = pd.DataFrame(self.episode_data_buffer)

            # 平均を計算
            avg_time_step = buffer_df['time_step'].mean()
            avg_reward = buffer_df['reward'].mean()
            avg_loss = buffer_df['loss'].mean()
            done_rate = buffer_df['done'].mean()

            # エピソード範囲を決定
            episode_group_start = self.episode_data_buffer[0]['episode']
            episode_group_end = self.episode_data_buffer[-1]['episode']

            # 平均をスコアサマリーファイル (scores_summary100.csv) にログ記録
            self._log_scores(episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss, done_rate)

            # バッファをクリアし、カウンターをリセット
            self.episode_data_buffer = []
            self.episode_data_counter = 0


    # 集計されたスコアをscores_summary100.csvにログ記録するように修正
    def _log_scores(self, episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss, done_rate):
        """
        集計された100エピソード分のスコアデータをscores_summary100.csvファイルに追記します。

        このメソッドはクラス内部でのみ使用されることを想定しています。

        Args:
            episode_group_start (int): 集計期間の開始エピソード番号。
            episode_group_end (int): 集計期間の終了エピソード番号。
            avg_time_step (float): 集計期間の平均ステップ数。
            avg_reward (float): 集計期間の平均報酬。
            avg_loss (float): 集計期間の平均損失。
            done_rate (float): 集計期間の成功率。
        """
        # 集計されたスコアデータをscores_summary100.csvファイルに直接追記
        with open(self.episode_batch_summary_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss, done_rate])


    # バッファに残っているスコアを保存するメソッドを追加 - 残りのエピソードデータを処理
    # このメソッドは、残りの個々のエピソードデータのみをscores_summary.csvに保存するようになりました (デバッグが必要な場合。それ以外は削除可能)
    def save_remaining_episode_data(self):
        """
        学習終了時などにバッファに残っているエピソードデータを集計し、
        scores_summary100.csvファイルに保存します。
        """
        if self.episode_data_buffer:# バッファに残ったデータを集計し、scores_summary100.csv に保存
            # 残りのバッファをDataFrameに変換
            buffer_df = pd.DataFrame(self.episode_data_buffer)

            # 残りのデータの平均を計算
            avg_time_step = buffer_df['time_step'].mean()
            avg_reward = buffer_df['reward'].mean()
            avg_loss = buffer_df['loss'].mean()
            done_rate = buffer_df['done'].mean()

            # 残りのデータのエピソード範囲を決定
            episode_group_start = self.episode_data_buffer[0]['episode']
            episode_group_end = self.episode_data_buffer[-1]['episode']

            # 残りのデータ用の平均をスコアサマリー100ファイルにログ記録
            self._log_scores(episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss, done_rate)

            print(f"Saved {len(self.episode_data_buffer)} remaining episode data entries as an aggregated group to {self.episode_batch_summary_path}")
            # バッファをクリアし、カウンターをリセット
            self.episode_data_buffer = []
            self.episode_data_counter = 0
        else:
            print("No remaining episode data to save.")


    # numpy配列を.npy形式で保存するように修正
    def save_visited_coordinates(self):
        """
        メモリ上のエージェント訪問回数グリッドを.npyファイルに保存します。
        保存後、訪問回数更新カウンターはリセットされます。
        """
        if self.visited_count_grid is not None:
            # numpy.saveを使用して配列を保存
            np.save(self.visited_coordinates_path, self.visited_count_grid)
            print(f"Visited count grid saved to {self.visited_coordinates_path}")
            # 保存後、訪問更新カウンターをリセット
            self.visited_updates_counter = 0
        else:
            print("No visited count grid to save.")

    def debug_print(self):
        """
        デバッグ用に現在の訪問回数グリッドをコンソールに出力します。
        """
        # リストではなく訪問回数グリッドを出力
        print(self.visited_count_grid)


import sqlite3

from typing import List, Tuple

CoordType = Tuple[int, int]


EpisodeTableType = tuple[int, bool, float, float]
"""
Episodeテーブルのフォーマット
(id, done, reward, loss)
"""

StepTableType = tuple[int, int]
"""
Stepテーブルのフォーマット
(episode_id, step_id)
"""

EnvTableType = tuple[int, int, int, str, int, int, int, int]
"""
Envテーブルのフォーマット (データベースから読み込む際に使用)
(id, episode_id, step_id, object_name, x_coord, y_coord, next_x_coord, next_y_coord)
"""

EnvInputType = tuple[int, int, str, int, int, int, int]
"""
Envテーブルへの挿入に使用するデータのフォーマット (id を除く)
(episode_id, step_id, object_name, x_coord, y_coord, next_x_coord, next_y_coord)
"""

AgentTableType = tuple[int, int, int, int, float]
"""
Agentテーブルのフォーマット
(id, episode_id, step_id, agent_id, action, reward)
"""

AgentInputType = tuple[int, int, str, int, float]
"""
Agentテーブルへの挿入に使用するデータのフォーマット (id を除く)
(episode_id, step_id, agent_id, action, reward)
"""

InputDataType = tuple[int, int, str, list[CoordType], list[CoordType], int, float]
"""
Env, Agentテーブルへの挿入に使用するデータのフォーマット
(episode_id, step_id, object_name, current_states, next_states, action, reward)
"""


class LoggerDB:

    BUFFER_FRECENCY = 10000

    def __init__(self, db_name="multi_agent_log.db"):
        self.db_name = db_name
        self._create_tables()

        self._insert_step_data:list[StepTableType] = []
        self._insert_buffer_env:list[EnvInputType] = []
        self._insert_buffer_agent:list[AgentInputType] = []
        self._insert_count = 0

    def _create_tables(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        c.execute("""
        CREATE TABLE IF NOT EXISTS Episodes (
            episode_id INT PRIMARY KEY,
            done BOOLEAN,       -- エピソードが終了したかどうか
            reward FLOAT,       -- エピソード報酬
            loss FLOAT          -- 損失(TD誤差であったり、nnの場合はモデルが返す値であったり)
        );
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS Steps (
            episode_id INT,
            step_id INT,
            PRIMARY KEY (episode_id, step_id),
            FOREIGN KEY (episode_id, step_id) REFERENCES Episodes(episode_id, step_id)
        );
        """)

        c.execute("""
        CREATE TABLE IF NOT EXISTS Env (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INT,     -- エピソードのID(episodesテーブルのidを参照)
            step_id INT,        -- ステップ番号(episodesテーブルのstep_idを参照)
            object_name VARCHAR(255), -- 'agent' or 'goal' or other
            x_coord INT,        -- オブジェクトのX座標
            y_coord INT,        -- オブジェクトのY座標
            next_x_coord INT,   -- オブジェクトの次状態のX座標
            next_y_coord INT,   -- オブジェクトの次状態のY座標
            FOREIGN KEY (episode_id,step_id) REFERENCES Episodes(episode_id,step_id)
        );
        """)

        c.execute("""CREATE TABLE IF NOT EXISTS Agents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id INT,     -- エピソードのID(episodesテーブルのidを参照)
            step_id INT,        -- ステップ番号(episodesテーブルのstep_idを参照)
            agent_id INT,       -- エージェントのID
            action INT,         -- 行動
            reward FLOAT,       -- 報酬
            FOREIGN KEY (episode_id,step_id) REFERENCES Episodes(episode_id,step_id)
        );
        """)

        conn.commit()
        conn.close()

    def insert_episode(self, episode_id: int, done: bool, reward: float, loss: float):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO Episodes (episode_id, done, reward, loss) VALUES (?, ?, ?, ?)", (episode_id, done, reward, loss))
            conn.commit()
        except sqlite3.IntegrityError:
            print(f"Episode with id {episode_id} already exists.")
        finally:
            conn.close()


    def insert_steps(self, buffer_step:list[StepTableType], buffer_env: list[EnvInputType], buffer_agent: list[AgentInputType]):
        if not buffer_env and not buffer_agent:
            return

        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        # c.execute("""SQL""")
        c.executemany("INSERT INTO Steps (episode_id, step_id) VALUES (?, ?)", buffer_step)
        c.executemany("INSERT INTO Env (episode_id, step_id, object_name, x_coord, y_coord, next_x_coord, next_y_coord) VALUES (?, ?, ?, ?, ?, ?, ?)", buffer_env)
        c.executemany("INSERT INTO Agents (episode_id, step_id, agent_id, action, reward) VALUES (?, ?, ?, ?, ?)", buffer_agent)

        conn.commit()
        conn.close()

    def insert_step(self, episode_id: int, step_id: int, object_data:dict, agents_data:dict):

        state_x, state_y = self._convert_state_to_db_format(object_data["cs"])
        next_state_x, next_state_y = self._convert_state_to_db_format(object_data["ns"])

        env_datas: list[EnvInputType] = [(episode_id, step_id, obname, cx, cy, nx, ny) for obname,cx,cy,nx,ny in zip(object_data["name"],state_x, state_y, next_state_x, next_state_y)]
        #print("env_datas: ",env_datas)
        agent_datas:list[AgentInputType] = [(episode_id, step_id, agent_id, action, reward) for agent_id,action,reward in zip(agents_data['id'],agents_data['action'],agents_data['reward'])]

        for env_data in env_datas:
            self._insert_buffer_env.append(env_data)

        for agent_data in agent_datas:
            self._insert_buffer_agent.append(agent_data)

        self._insert_step_data.append((episode_id, step_id))
        self._insert_count += 1

        #print("_insert_buffer_env: ",self._insert_buffer_env)

        if self._insert_count >= self.BUFFER_FRECENCY:
            self.insert_steps(self._insert_step_data, self._insert_buffer_env, self._insert_buffer_agent)
            self._insert_buffer_env.clear()
            self._insert_buffer_agent.clear()
            self._insert_step_data.clear()
            self._insert_count = 0

    def _convert_state_to_db_format(self, state: list[CoordType]):
        # Assuming the format is [goal1, goal2, agent1, agent2, ...]

        state_x = [coord[0] for coord in state]
        state_y = [coord[1] for coord in state]

        return state_x, state_y

    def _decode_step_state(self, x:list,y:list):
        state = [(x[i],y[i]) for i in range(len(x))]
        return state

    def flush_buffer(self):
        if not self._insert_buffer_env and not self._insert_buffer_agent:
            return

        self.insert_steps(self._insert_step_data, self._insert_buffer_env, self._insert_buffer_agent)
        self._insert_buffer_env.clear()
        self._insert_buffer_agent.clear()
        self._insert_count = 0

    def get_episode_data(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        c.execute("SELECT * FROM Episodes")
        episode_data:list[EpisodeTableType] = c.fetchall()

        conn.close()

        return episode_data

    def get_step_data(self, episode_id: int):
        if episode_id < 1:
            raise ValueError(f"episodeの値は正にしてほしい。現在: {episode_id}")

        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        episode_data_env:list[EnvTableType] = []
        c.execute("SELECT * FROM Env WHERE episode_id = ?", (episode_id,))
        episode_data_env = c.fetchall()

        c.execute("SELECT * FROM Agents WHERE episode_id = ?", (episode_id,))
        episode_data_agent:list[AgentTableType] = c.fetchall()
        conn.close()

        return episode_data_env, episode_data_agent


    def end_episode(self, episode_id: int, done: bool, reward: float, loss: float):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("UPDATE Episodes SET done = ?, reward = ?, loss = ? WHERE episode_id = ?", (done, reward, loss, episode_id))
        conn.commit()
        conn.close()

    def __len__(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM Steps")
        count = c.fetchone()[0]
        conn.close()
        return count
