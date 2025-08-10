import numpy as np
import os
import csv
import pandas as pd
import torch

class Saver:
    """
    学習中に発生する様々なデータをファイルに保存するためのクラス.

    エピソードごとのスコア、エージェントの訪問座標、モデルの重みなどを管理し、
    指定されたディレクトリに保存します。
    エピソードスコアは100エピソードごとに集計して保存する機能も持ちます。
    """
    def __init__(self, save_dir, grid_size):
        """
        Saverクラスの新しいインスタンスを初期化します。

        Args:
            save_dir (str): データを保存するディレクトリのパス。
            grid_size (int): エージェントの訪問座標を記録するグリッドのサイズ (NxN)。
        """
        self.save_dir = save_dir
        self.grid_size = grid_size # グリッドサイズを保存
        self.scores_summary_path = os.path.join(self.save_dir, "scores_summary.csv")
        # 100エピソードサマリー用の新しいパス
        self.scores_summary_100_path = os.path.join(self.save_dir, "scores_summary100.csv")
        # ヒートマップデータ用にファイル名と形式を.npyに変更
        self.visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")
        #self.model_dir_path = os.path.join(self.save_dir, 'model_weights')

        os.makedirs(self.save_dir, exist_ok=True)
        #os.makedirs(self.model_dir_path, exist_ok=True)

        # scores_summary.csvを作成 (集計前の個々のエピソードデータ用バッファ)
        if not os.path.exists(self.scores_summary_path):
            # scores_summary.csvのヘッダーは、生のエピソードデータを一時的に保存し、
            # 集計されたデータがscores_summary100.csvに書き込まれるため、もはや不要です。
            pass # この一時バッファファイルにはヘッダーは不要

        # scores_summary100.csvを作成 (集計データ用)
        if not os.path.exists(self.scores_summary_100_path):
            with open(self.scores_summary_100_path, "w", newline='') as f:
                # 100エピソードサマリーファイル用の更新されたヘッダー
                csv.writer(f).writerow(["episode_group_start", "episode_group_end", "avg_time_step_100", "avg_reward_100", "avg_loss_100"])


        # 訪問回数を記録するためのNxN numpy配列としてインメモリストレージを初期化
        # ゼロで初期化
        self.visited_count_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # 個々のエピソードデータ用バッファ
        self.episode_data_buffer = []
        # 個々のエピソードデータ用カウンター
        self.episode_data_counter = 0


        # visited_count_gridは訪問座標用のインメモリバッファ
        # 訪問座標更新カウンターを追加
        self.visited_updates_counter = 0


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
        # xとyがグリッドの境界内にあると仮定
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            # 訪問したセルのカウントをインクリメント
            self.visited_count_grid[y, x] += 1
            # 訪問更新カウンターをインクリメント
            self.visited_updates_counter += 1
        else:
            print(f"警告: エージェント状態座標が無効です: x={x}, y={y} (エージェントID: {agent_id})")


    # 個々のエピソードデータを記録し、平均のログ記録をトリガーする新しいメソッド
    def log_episode_data(self, episode: int, time_step: int, reward: float, loss: float):
        """
        個々のエピソードデータをバッファに記録し、100エピソードごとに集計してファイルに保存します。

        Args:
            episode (int): エピソード番号。
            time_step (int): そのエピソードでのステップ数。
            reward (float): そのエピソードで得られた報酬の合計。
            loss (float): そのエピソードでの学習損失 (該当する場合)。
        """
        # 個々のエピソードデータをバッファに追加
        self.episode_data_buffer.append({'episode': episode, 'time_step': time_step, 'reward': reward, 'loss': loss})
        # カウンターをインクリメント
        self.episode_data_counter += 1

        # カウンターが100の倍数で、かつ0より大きいかチェック
        if self.episode_data_counter % 100 == 0 and self.episode_data_counter > 0:
            # バッファをDataFrameに変換
            buffer_df = pd.DataFrame(self.episode_data_buffer)

            # 平均を計算
            avg_time_step = buffer_df['time_step'].mean()
            avg_reward = buffer_df['reward'].mean()
            avg_loss = buffer_df['loss'].mean()

            # エピソード範囲を決定
            episode_group_start = self.episode_data_buffer[0]['episode']
            episode_group_end = self.episode_data_buffer[-1]['episode']

            # 平均をスコアサマリーファイル (scores_summary100.csv) にログ記録
            self._log_scores(episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss)

            # バッファをクリアし、カウンターをリセット
            self.episode_data_buffer = []
            self.episode_data_counter = 0


    # 集計されたスコアをscores_summary100.csvにログ記録するように修正
    def _log_scores(self, episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss):
        """
        集計された100エピソード分のスコアデータをscores_summary100.csvファイルに追記します。

        このメソッドはクラス内部でのみ使用されることを想定しています。

        Args:
            episode_group_start (int): 集計期間の開始エピソード番号。
            episode_group_end (int): 集計期間の終了エピソード番号。
            avg_time_step (float): 集計期間の平均ステップ数。
            avg_reward (float): 集計期間の平均報酬。
            avg_loss (float): 集計期間の平均損失。
        """
        # 集計されたスコアデータをscores_summary100.csvファイルに直接追記
        with open(self.scores_summary_100_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss])


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

            # 残りのデータのエピソード範囲を決定
            episode_group_start = self.episode_data_buffer[0]['episode']
            episode_group_end = self.episode_data_buffer[-1]['episode']

            # 残りのデータ用の平均をスコアサマリー100ファイルにログ記録
            self._log_scores(episode_group_start, episode_group_end, avg_time_step, avg_reward, avg_loss)

            print(f"Saved {len(self.episode_data_buffer)} remaining episode data entries as an aggregated group to {self.scores_summary_100_path}")
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