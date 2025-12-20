import unittest
import os
import shutil
import numpy as np
import pandas as pd
import csv
import io
from contextlib import redirect_stdout

from src.utils.Saver import Saver

class TestSaver(unittest.TestCase):
    """
    ### `Saver`クラス単体テストの概要

この単体テストは、`Saver`クラスが意図通りに動作することを確認するために実装されています。主要なテスト項目は以下の通りです。

*   **初期化 (`test_init`)**: `Saver`インスタンスの初期化が正しく行われるか、必要なファイル（`scores_summary100.csv`）が作成され、正しいヘッダーが書き込まれているか、`visited_count_grid`が適切に初期化されているかを確認します。
*   **エージェントの状態ロギング (`test_log_agent_states`)**: エージェントの訪問座標が`visited_count_grid`に正しく記録され、無効な座標が渡された場合に`ValueError`が発生するかを検証します。
*   **訪問座標の保存 (`test_save_visited_coordinates`)**: メモリ上の`visited_count_grid`が`.npy`ファイルとして正しく保存され、保存後に`visited_updates_counter`がリセットされるかを確認します。
*   **エピソードデータの集計と保存 (`test_log_episode_data_aggregation`)**: エピソードデータがバッファリングされ、`CALCULATION_PERIOD`（この場合は100エピソード）ごとに集計されて`scores_summary100.csv`に保存されるか、計算された平均値が正しいかを検証します。
*   **残りのエピソードデータの保存 (`test_save_remaining_episode_data`)**: 学習終了時にバッファに残ったエピソードデータが正しく集計され、`scores_summary100.csv`に保存されるかを確認します。また、データがない場合に`save_remaining_episode_data`を呼び出しても問題ないことも確認します。
*   **ダミー実装のテスト (`test_q_table_and_dqn_weights_placeholders`)**: `save_q_table`と`save_dqn_weights`が、期待されるプレースホルダーメッセージをコンソールに出力することを確認します。
    """

    def setUp(self):
        """
        各テストケースの前に実行され、一時的な保存ディレクトリとSaverインスタンスを作成します。
        """
        self.test_dir = "./test_save_dir"
        self.grid_size = 5
        os.makedirs(self.test_dir, exist_ok=True)

        # position_validator_funcをラムダ関数で定義して注入
        self.position_validator = lambda pos: 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size
        self.saver = Saver(self.test_dir, self.grid_size, self.position_validator)

    def tearDown(self):
        """
        各テストケースの後に実行され、一時的な保存ディレクトリを削除します。
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init(self):
        """
        Saverクラスの初期化が正しく行われることをテストします。
        """
        # 保存ディレクトリが存在することを確認
        self.assertTrue(os.path.exists(self.test_dir))

        # scores_summary100.csvが作成され、正しいヘッダーを持つことを確認
        scores_100_path = os.path.join(self.test_dir, f"aggregated_episode_metrics_{Saver.CALCULATION_PERIOD}.csv")
        self.assertTrue(os.path.exists(scores_100_path))
        with open(scores_100_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["episode_group_start", "episode_group_end", "avg_time_step_100", "avg_reward_100", "avg_loss_100", "done_rate"])

        # visited_count_gridが正しく初期化されていることを確認
        self.assertIsInstance(self.saver.visited_count_grid, np.ndarray)
        self.assertEqual(self.saver.visited_count_grid.shape, (self.grid_size, self.grid_size))
        self.assertTrue(np.all(self.saver.visited_count_grid == 0))
        self.assertEqual(self.saver.visited_updates_counter, 0)

        # 注入されたバリデーターが設定されていることを確認
        self.assertEqual(self.saver.is_position_valid, self.position_validator)

    def test_log_agent_states(self):
        """
        エージェントの訪問座標が正しく記録されることをテストします。
        """
        self.saver.log_agent_states(agent_id=0, x=0, y=0)
        self.assertEqual(self.saver.visited_count_grid[0, 0], 1)
        self.assertEqual(self.saver.visited_updates_counter, 1)

        self.saver.log_agent_states(agent_id=0, x=1, y=1)
        self.saver.log_agent_states(agent_id=1, x=1, y=1)
        self.assertEqual(self.saver.visited_count_grid[1, 1], 2)
        self.assertEqual(self.saver.visited_updates_counter, 3)

        # 無効な座標の場合にValueErrorが発生することを確認 (注入されたバリデーターによって)
        with self.assertRaises(ValueError):
            self.saver.log_agent_states(agent_id=0, x=self.grid_size, y=0)
        with self.assertRaises(ValueError):
            self.saver.log_agent_states(agent_id=0, x=0, y=self.grid_size)

    def test_save_visited_coordinates(self):
        """
        訪問座標グリッドが.npyファイルとして正しく保存されることをテストします。
        """
        self.saver.log_agent_states(agent_id=0, x=0, y=0)
        self.saver.log_agent_states(agent_id=0, x=1, y=1)
        initial_grid = np.copy(self.saver.visited_count_grid)

        self.saver.save_visited_coordinates()

        # ファイルが存在することを確認
        npy_path = os.path.join(self.test_dir, "visited_coordinates.npy")
        self.assertTrue(os.path.exists(npy_path))

        # 保存されたデータが正しいことを確認
        loaded_grid = np.load(npy_path)
        self.assertTrue(np.array_equal(initial_grid, loaded_grid))

        # 訪問更新カウンターがリセットされていることを確認
        self.assertEqual(self.saver.visited_updates_counter, 0)

    def test_log_episode_data_aggregation(self):
        """
        エピソードデータがバッファリングされ、CALCULATION_PERIODごとに集計・保存されることをテストします。
        """
        num_episodes = self.saver.CALCULATION_PERIOD * 2 + 5 # 2回集計 + 残り5エピソード
        expected_time_steps = []
        expected_rewards = []
        expected_losses = []
        expected_dones = []

        for i in range(num_episodes):
            time_step = 10 + i % 5 # 少し異なる値を生成
            reward = 1.0 * (i % 2) # 0.0 or 1.0
            loss = 0.1 * (i % 3) # 0.0, 0.1, 0.2
            done = (i % 2 == 0) # True or False
            self.saver.log_episode_data(episode=i, time_step=time_step, reward=reward, loss=loss, done=done)
            expected_time_steps.append(time_step)
            expected_rewards.append(reward)
            expected_losses.append(loss)
            expected_dones.append(int(done))

        scores_100_path = os.path.join(self.test_dir, f"aggregated_episode_metrics_{Saver.CALCULATION_PERIOD}.csv")
        df = pd.read_csv(scores_100_path)

        # 2つの集計グループが保存されていることを確認
        self.assertEqual(len(df), 2)

        # 最初のグループの検証 (エピソード0 - CALCULATION_PERIOD-1)
        start_idx_0 = 0
        end_idx_0 = self.saver.CALCULATION_PERIOD
        self.assertEqual(df.loc[0, 'episode_group_start'], 0)
        self.assertEqual(df.loc[0, 'episode_group_end'], self.saver.CALCULATION_PERIOD - 1)
        self.assertAlmostEqual(df.loc[0, 'avg_time_step_100'], float(np.mean(expected_time_steps[start_idx_0:end_idx_0])))  # type:ignore
        self.assertAlmostEqual(df.loc[0, 'avg_reward_100'], float(np.mean(expected_rewards[start_idx_0:end_idx_0])))        # type:ignore
        self.assertAlmostEqual(df.loc[0, 'avg_loss_100'], float(np.mean(expected_losses[start_idx_0:end_idx_0])))           # type:ignore
        self.assertAlmostEqual(df.loc[0, 'done_rate'], float(np.mean(expected_dones[start_idx_0:end_idx_0])))               # type:ignore

        # 2番目のグループの検証 (エピソードCALCULATION_PERIOD - 2*CALCULATION_PERIOD-1)
        start_idx_1 = self.saver.CALCULATION_PERIOD
        end_idx_1 = self.saver.CALCULATION_PERIOD * 2
        self.assertEqual(df.loc[1, 'episode_group_start'], self.saver.CALCULATION_PERIOD)
        self.assertEqual(df.loc[1, 'episode_group_end'], self.saver.CALCULATION_PERIOD * 2 - 1)
        self.assertAlmostEqual(df.loc[1, 'avg_time_step_100'], float(np.mean(expected_time_steps[start_idx_1:end_idx_1])))# type:ignore
        self.assertAlmostEqual(df.loc[1, 'avg_reward_100'], float(np.mean(expected_rewards[start_idx_1:end_idx_1])))# type:ignore
        self.assertAlmostEqual(df.loc[1, 'avg_loss_100'], float(np.mean(expected_losses[start_idx_1:end_idx_1])))# type:ignore
        self.assertAlmostEqual(df.loc[1, 'done_rate'], float(np.mean(expected_dones[start_idx_1:end_idx_1])))# type:ignore

        # バッファに残りがあることを確認
        self.assertEqual(len(self.saver.episode_data_buffer), 5)
        self.assertEqual(self.saver.episode_data_counter, 5)

    def test_save_remaining_episode_data(self):
        """
        残りのエピソードデータが学習終了時に正しく保存されることをテストします。
        """
        num_remaining = 3
        expected_time_steps = []
        expected_rewards = []
        expected_losses = []
        expected_dones = []

        for i in range(num_remaining):
            time_step = 20 + i
            reward = 0.5 * i
            loss = 0.05 * i
            done = (i == 2)
            self.saver.log_episode_data(episode=i, time_step=time_step, reward=reward, loss=loss, done=done)
            expected_time_steps.append(time_step)
            expected_rewards.append(reward)
            expected_losses.append(loss)
            expected_dones.append(int(done))

        # 集計期間に満たないため、ファイルはまだ空
        scores_100_path = os.path.join(self.test_dir, f"aggregated_episode_metrics_{Saver.CALCULATION_PERIOD}.csv")
        initial_df = pd.read_csv(scores_100_path)
        self.assertEqual(len(initial_df), 0)

        self.saver.save_remaining_episode_data()

        # 残りのデータが保存されたことを確認
        final_df = pd.read_csv(scores_100_path)
        self.assertEqual(len(final_df), 1)

        # 保存されたデータが正しいことを確認
        self.assertEqual(final_df.loc[0, 'episode_group_start'], 0)
        self.assertEqual(final_df.loc[0, 'episode_group_end'], num_remaining - 1)
        self.assertAlmostEqual(final_df.loc[0, 'avg_time_step_100'], float(np.mean(expected_time_steps)))# type:ignore
        self.assertAlmostEqual(final_df.loc[0, 'avg_reward_100'], float(np.mean(expected_rewards)))# type:ignore
        self.assertAlmostEqual(final_df.loc[0, 'avg_loss_100'], float(np.mean(expected_losses)))# type:ignore
        self.assertAlmostEqual(final_df.loc[0, 'done_rate'], float(np.mean(expected_dones)))# type:ignore

        # バッファとカウンターがリセットされていることを確認
        self.assertEqual(len(self.saver.episode_data_buffer), 0)
        self.assertEqual(self.saver.episode_data_counter, 0)

    def test_save_remaining_episode_data_no_data(self):
        """
        バッファにデータがない場合に`save_remaining_episode_data`を呼び出すことをテストします。
        """
        # データが何もログされていないことを確認
        self.assertEqual(len(self.saver.episode_data_buffer), 0)

        # 呼び出し時にエラーが発生しないこと、ファイルが変更されないことを確認
        scores_100_path = os.path.join(self.test_dir, f"aggregated_episode_metrics_{Saver.CALCULATION_PERIOD}.csv")
        initial_df = pd.read_csv(scores_100_path)

        self.saver.save_remaining_episode_data()

        final_df = pd.read_csv(scores_100_path)
        self.assertEqual(len(initial_df), len(final_df))
        self.assertTrue(initial_df.equals(final_df))


    def test_q_table_and_dqn_weights_placeholders(self):
        """
        QテーブルとDQN重み保存メソッドが正しいメッセージを出力することをテストします（ダミー実装）。
        """
        # save_q_table
        f = io.StringIO()
        with redirect_stdout(f):
            self.saver.save_q_table(agents=None, mask=None)
        self.assertEqual(f.getvalue().strip(), 'Qテーブル保存プロセスはModel_IOに移行')

        # save_dqn_weights
        f = io.StringIO()
        with redirect_stdout(f):
            self.saver.save_dqn_weights(agents=None)
        self.assertEqual(f.getvalue().strip(), 'モデル重み保存プロセスはModel_IOに移行')

