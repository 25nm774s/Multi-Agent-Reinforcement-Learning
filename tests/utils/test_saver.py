import unittest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.utils.Model_Saver import Saver

class TestSaver(unittest.TestCase):
    """Saverクラスの単体テスト."""

    def setUp(self):
        """各テストメソッド実行前に呼ばれるセットアップ処理."""
        self.save_dir = "."
        self.grid_size = 10 # グリッドサイズを定義

        # Saverインスタンスを作成する前にクリーンアップ用のファイルパスを定義
        self.scores_summary_path = os.path.join(self.save_dir, "scores_summary.csv")
        self.scores_summary_100_path = os.path.join(self.save_dir, "scores_summary100.csv") # 100エピソードサマリー用の新しいパス
        self.visited_coordinates_path = os.path.join(self.save_dir, "visited_coordinates.npy")

        # Saverインスタンスを作成する前に前回の実行で作成されたファイルをクリーンアップ
        if os.path.exists(self.scores_summary_path):
            os.remove(self.scores_summary_path)
            # print(f"setUp: Removed {self.scores_summary_path}") # デバッグプリント削除
        if os.path.exists(self.scores_summary_100_path): # 新しいファイルをクリーンアップ
            os.remove(self.scores_summary_100_path)
            # print(f"setUp: Removed {self.scores_summary_100_path}") # デバッグプリント削除
        if os.path.exists(self.visited_coordinates_path):
            os.remove(self.visited_coordinates_path)
            # print(f"setUp: Removed {self.visited_coordinates_path}") # デバッグプリント削除

        self.saver = Saver(self.save_dir, self.grid_size)
        # print(f"setUp: Saver instance created. scores_summary_100_path exists: {os.path.exists(self.scores_summary_100_path)}") # デバッグプリント削除


    def tearDown(self):
        """各テストメソッド実行後に呼ばれるクリーンアップ処理."""
        # print(f"tearDown: Cleaning up files in {self.save_dir}") # デバッグプリント削除
        # Saverインスタンスによって作成されたファイルを明示的に削除
        if os.path.exists(self.scores_summary_path):
            os.remove(self.scores_summary_path)
            # print(f"tearDown: Removed {self.scores_summary_path}") # デバッグプリント削除
        if os.path.exists(self.scores_summary_100_path):
            os.remove(self.scores_summary_100_path)
            # print(f"tearDown: Removed {self.scores_summary_100_path}") # デバッグプリント削除
        if os.path.exists(self.visited_coordinates_path):
            os.remove(self.visited_coordinates_path)
            # print(f"tearDown: Removed {self.visited_coordinates_path}") # デバッグプリント削除
        # print("tearDown: Cleanup complete.") # デバッグプリント削除


    def test_saver_log_episode_data(self):
        """log_episode_data が個々のエピソードデータをバッファリングし、100コールごとに平均を正しく計算してログに記録するかをテスト."""
        # マングルされた名前を使用してクラスのメソッドをパッチ
        with patch('__main__.Saver._log_scores') as mock_log_scores:
            # 99エピソードをログ記録
            for i in range(1, 100):
                self.saver.log_episode_data(i, i * 10, i * 0.1, i * 0.01)

            self.assertEqual(len(self.saver.episode_data_buffer), 99)
            mock_log_scores.assert_not_called()

            # 100番目のエピソードをログ記録。これによりlog_scoresがトリガーされるはず
            self.saver.log_episode_data(100, 1000, 10.0, 1.0)

            self.assertEqual(len(self.saver.episode_data_buffer), 0)
            mock_log_scores.assert_called_once()

            # 期待される平均を手動で計算
            buffer_df = pd.DataFrame([{'episode': i, 'time_step': i * 10, 'reward': i * 0.1, 'loss': i * 0.01} for i in range(1, 100)] + [{'episode': 100, 'time_step': 1000, 'reward': 10.0, 'loss': 1.0}])
            expected_avg_time_step = buffer_df['time_step'].mean()
            expected_avg_reward = buffer_df['reward'].mean()
            expected_avg_loss = buffer_df['loss'].mean()

            call_args, _ = mock_log_scores.call_args
            self.assertEqual(call_args[0], 1)
            self.assertEqual(call_args[1], 100)
            self.assertAlmostEqual(call_args[2], expected_avg_time_step)
            self.assertAlmostEqual(call_args[3], expected_avg_reward)
            self.assertAlmostEqual(call_args[4], expected_avg_loss)

    def test_saver_save_remaining_episode_data(self):
        """save_remaining_episode_data が残りのバッファデータを正しくファイルに書き込むかテスト."""
        saver = self.saver

        # このテストではファイル内容を確認するため、_log_scores をモックしない

        # 100エピソード未満をログ記録
        num_episodes_to_log = 42
        for i in range(1, num_episodes_to_log + 1):
            saver.log_episode_data(i, i * 10, i * 0.1, i * 0.01)

        self.assertEqual(len(saver.episode_data_buffer), num_episodes_to_log)

        # 残りのエピソードデータを保存 - これにより内部的に_log_scoresがトリガーされるはず
        saver.save_remaining_episode_data()

        self.assertEqual(len(saver.episode_data_buffer), 0)


        # scores_summary100.csv ファイルを読み込み、内容を確認
        self.assertTrue(os.path.exists(saver.scores_summary_100_path))
        with open(saver.scores_summary_100_path, "r") as f:
            content = f.readlines()

        # ヘッダー + 残りの集計データ用の1データ行
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0].strip(), "episode_group_start,episode_group_end,avg_time_step_100,avg_reward_100,avg_loss_100")


        # scores_summary100.csv のログに記録された値を確認
        logged_values = content[1].strip().split(',')

        # 残りのデータに対する期待される平均を手動で計算
        buffer_df = pd.DataFrame([{'episode': i, 'time_step': i * 10, 'reward': i * 0.1, 'loss': i * 0.01} for i in range(1, num_episodes_to_log + 1)])
        expected_avg_time_step = buffer_df['time_step'].mean()
        expected_avg_reward = buffer_df['reward'].mean()
        expected_avg_loss = buffer_df['loss'].mean()

        self.assertEqual(int(logged_values[0]), 1) # episode_group_start
        self.assertEqual(int(logged_values[1]), num_episodes_to_log) # episode_group_end
        self.assertAlmostEqual(float(logged_values[2]), expected_avg_time_step) # avg_time_step
        self.assertAlmostEqual(float(logged_values[3]), expected_avg_reward) # avg_reward
        self.assertAlmostEqual(float(logged_values[4]), expected_avg_loss) # avg_loss

        # 元の scores_summary.csv が空または作成されていないことを確認 (現在はバッファリング用)
        self.assertFalse(os.path.exists(saver.scores_summary_path) and os.stat(saver.scores_summary_path).st_size > 0)


    def test_saver_log_agent_states_in_memory(self):
        """log_agent_states がエージェント状態をメモリのNxNグリッドに正しく格納し、カウンターをインクリメントするかテスト."""
        saver = self.saver
        # visited_count_grid は setUp でゼロ初期化される
        self.assertEqual(saver.visited_updates_counter, 0) # 初期カウンターを確認

        saver.log_agent_states(0, 5, 10) # y=10は10x10グリッド(0-9)の範囲外なので警告が表示されるはず
        self.assertEqual(saver.visited_updates_counter, 0) # 無効な座標の場合はカウンターはインクリメントされない

        saver.log_agent_states(1, 2, 3)
        self.assertEqual(saver.visited_updates_counter, 1)

        saver.log_agent_states(0, 5, 5)
        self.assertEqual(saver.visited_updates_counter, 2)

        saver.log_agent_states(1, 2, 3) # 同じセルを再度訪問
        self.assertEqual(saver.visited_updates_counter, 3)

        # visited_count_grid を直接確認
        expected_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        expected_grid[3, 2] = 2 # (x=2, y=3) を2回訪問
        expected_grid[5, 5] = 1 # (x=5, y=5) を1回訪問

        # numpy配列の比較には np.testing.assert_array_equal を使用
        np.testing.assert_array_equal(saver.visited_count_grid, expected_grid)


    def test_saver_save_visited_coordinates(self):
        """save_visited_coordinates がメモリ上のNxNグリッドを.npyファイルに正しく保存し、カウンターをリセットするかテスト."""
        saver = self.saver
        # メモリ上の visited_count_grid を設定し、カウンターをインクリメント
        saver.log_agent_states(0, 1, 1)
        saver.log_agent_states(1, 3, 4)
        saver.log_agent_states(0, 1, 1)
        self.assertEqual(saver.visited_updates_counter, 3)

        # 保存先ファイルが保存前にクリーンであることを確認
        if os.path.exists(saver.visited_coordinates_path):
            os.remove(saver.visited_coordinates_path)

        saver.save_visited_coordinates()

        self.assertTrue(os.path.exists(saver.visited_coordinates_path))
        self.assertEqual(saver.visited_updates_counter, 0) # カウンターはリセットされるはず

        # numpy.load を使用して保存されたデータをロード
        loaded_grid = np.load(saver.visited_coordinates_path)

        # ロードされたグリッドが元のグリッド (saver.visited_count_grid にある) と等しいか確認
        expected_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        expected_grid[1, 1] = 2
        expected_grid[4, 3] = 1
        np.testing.assert_array_equal(loaded_grid, expected_grid)

    def test_saver_log_episode_data_to_100_summary(self):
        """log_episode_data が100エピソードごとにscores_summary100.csvに集計データを正しく書き込むかテスト."""
        saver = self.saver
        num_episodes_to_log = 250 # 複数(2つ)の100エピソードグループと残りのデータをテストするために250エピソードをログ記録

        # このテストではファイル内容を確認するため、_log_scores をモックしない

        # エピソードデータをログ記録
        for i in range(1, num_episodes_to_log + 1):
            saver.log_episode_data(i, i * 10, i * 0.1, i * 0.01)

        # _log_scoresが呼び出された回数を確認 (2つの完全な100エピソードグループで2回呼び出されるはず)
        # ファイル内容を直接確認する

        # 残りのエピソードデータ (最後の50エピソード) を保存 - これにより内部的に_log_scoresが1回トリガーされるはず
        saver.save_remaining_episode_data()

        self.assertEqual(len(saver.episode_data_buffer), 0)
        # __log_scoresの合計呼び出し回数はここではassertできない
        # ファイル内容を直接確認する


        # scores_summary100.csv ファイルを読み込み
        self.assertTrue(os.path.exists(saver.scores_summary_100_path))
        with open(saver.scores_summary_100_path, "r") as f:
            content = f.readlines()

        # 期待される行数: ヘッダー + 2つの完全な100エピソードグループ + 1つの残りのグループ
        self.assertEqual(len(content), 4)
        self.assertEqual(content[0].strip(), "episode_group_start,episode_group_end,avg_time_step_100,avg_reward_100,avg_loss_100")

        # ファイル内容から最初の100エピソードグループ (エピソード1-100) を確認
        logged_values_group1 = content[1].strip().split(',')
        self.assertEqual(int(logged_values_group1[0]), 1)
        self.assertEqual(int(logged_values_group1[1]), 100)
        # エピソード1-100の期待される平均を計算
        df_group1 = pd.DataFrame([{'episode': i, 'time_step': i * 10, 'reward': i * 0.1, 'loss': i * 0.01} for i in range(1, 101)])
        self.assertAlmostEqual(float(logged_values_group1[2]), df_group1['time_step'].mean())
        self.assertAlmostEqual(float(logged_values_group1[3]), df_group1['reward'].mean())
        self.assertAlmostEqual(float(logged_values_group1[4]), df_group1['loss'].mean())

        # ファイル内容から2番目の100エピソードグループ (エピソード101-200) を確認
        logged_values_group2 = content[2].strip().split(',')
        self.assertEqual(int(logged_values_group2[0]), 101)
        self.assertEqual(int(logged_values_group2[1]), 200)
        # エピソード101-200の期待される平均を計算
        df_group2 = pd.DataFrame([{'episode': i, 'time_step': i * 10, 'reward': i * 0.1, 'loss': i * 0.01} for i in range(101, 201)])
        self.assertAlmostEqual(float(logged_values_group2[2]), df_group2['time_step'].mean())
        self.assertAlmostEqual(float(logged_values_group2[3]), df_group2['reward'].mean())
        self.assertAlmostEqual(float(logged_values_group2[4]), df_group2['loss'].mean())

        # ファイル内容から残りのエピソードグループ (エピソード201-250) を確認
        logged_values_group3 = content[3].strip().split(',')
        self.assertEqual(int(logged_values_group3[0]), 201)
        self.assertEqual(int(logged_values_group3[1]), 250)
        # エピソード201-250の期待される平均を計算
        df_group3 = pd.DataFrame([{'episode': i, 'time_step': i * 10, 'reward': i * 0.1, 'loss': i * 0.01} for i in range(201, 251)])
        self.assertAlmostEqual(float(logged_values_group3[2]), df_group3['time_step'].mean())
        self.assertAlmostEqual(float(logged_values_group3[3]), df_group3['reward'].mean())
        self.assertAlmostEqual(float(logged_values_group3[4]), df_group3['loss'].mean())

        # 元の scores_summary.csv が空または作成されていないことを確認
        self.assertFalse(os.path.exists(saver.scores_summary_path) and os.stat(saver.scores_summary_path).st_size > 0)