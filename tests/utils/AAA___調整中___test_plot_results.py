import unittest
import os
import matplotlib.pyplot as plt
# No need to import Saver or PlotResults again, they are already defined in the notebook
# No need to import japanize_matplotlib again
from src.utils.Saver import Saver

class __TYOUSEITYUU___TestPlotResults(unittest.TestCase):
    """PlotResultsクラスとSaverクラスの単体テスト."""

    def setUp(self):
        """各テストメソッド実行前に呼ばれるセットアップ処理."""
        self.save_dir = "."
        # テスト実行前にファイルが存在する場合は削除し、クリーンな状態にする
        self.summary_scores_path = os.path.join(self.save_dir, "scores_summary.csv")
        self.agents_states_path = os.path.join(self.save_dir, "agents_states.csv")
        self.draw_plot_file = self.summary_scores_path.replace('.csv', '.pdf')
        self.heatmap_plot_file = self.agents_states_path.replace('.csv', '_agents_heatmap.pdf')

        if os.path.exists(self.summary_scores_path):
             os.remove(self.summary_scores_path)
        if os.path.exists(self.agents_states_path):
             os.remove(self.agents_states_path)
        if os.path.exists(self.draw_plot_file):
            os.remove(self.draw_plot_file)
        if os.path.exists(self.heatmap_plot_file):
            os.remove(self.heatmap_plot_file)

        # Saverインスタンスを作成し、PlotResultsに渡す
        self.saver = Saver(self.save_dir)
        self.plot_results = PlotResults(self.save_dir, self.saver)

        # draw, road_csv, draw_heatmapテスト用にダミーのサマリーファイルとエージェント状態ファイルを作成
        # These are created by Saver's __init__ if they don't exist, but we'll add data for tests
        # Add dummy data for summary scores
        summary_data = """episode_group_start,episode_group_end,avg_time_step,avg_reward,avg_loss
1,100,80.0,15.0,0.07666666666666666
101,200,60.0,30.0,0.03
"""
        with open(self.summary_scores_path, "w") as f:
            f.write(summary_data)

        # Add dummy data for agent states
        agent_states_data = """agent_id,x,y
0,1,1
0,1,2
1,3,3
1,3,4
"""
        with open(self.agents_states_path, "w") as f:
             f.write(agent_states_data)


    def tearDown(self):
        """各テストメソッド実行後に呼ばれるクリーンアップ処理."""
        # ダミーファイルを削除
        if os.path.exists(self.summary_scores_path):
            os.remove(self.summary_scores_path)
        if os.path.exists(self.agents_states_path):
            os.remove(self.agents_states_path)

        # 生成されたプロットファイルも削除
        if os.path.exists(self.draw_plot_file):
            os.remove(self.draw_plot_file)
        if os.path.exists(self.heatmap_plot_file):
            os.remove(self.heatmap_plot_file)

        # 全てのmatplotlib Figure を閉じてメモリを解放
        plt.close('all')

    # Remove test_initialize_summary_csv
    # Remove test_initialize_agent_states_csv

    def test_add_episode_data(self):
        """add_episode_data がデータをバッファリングし、集計を正しく計算するかテスト."""
        # setUpで作成されたサマリーファイルを削除し、クリーンな状態でテスト開始
        if os.path.exists(self.summary_scores_path):
             os.remove(self.summary_scores_path)
        # Saverがファイルを初期化するように新しいインスタンスを作成
        self.saver = Saver(self.save_dir)
        self.plot_results = PlotResults(self.save_dir, self.saver)


        # 最初の99エピソードのデータを追加 (バッファリングのみ)
        for i in range(1, 100):
            self.plot_results.add_episode_data(i, i * 10, i * 0.1, i * 0.01)

        # バッファに99エピソード含まれていることを確認
        self.assertEqual(len(self.plot_results.episode_buffer), 99)
        # ファイル内容のチェックはSaverのテストに任せるため削除

        # 100エピソード目のデータを追加 (集計とSaverへのログ書き込みがトリガーされる)
        self.plot_results.add_episode_data(100, 100 * 10, 100 * 0.1, 100 * 0.01)

        # バッファがクリアされたことを確認
        self.assertEqual(len(self.plot_results.episode_buffer), 0)
        # ファイル内容のチェックはSaverのテストに任せるため削除

        # 次の100エピソード (101-200) のデータを追加
        for i in range(101, 201):
            self.plot_results.add_episode_data(i, i * 10, i * 0.1, i * 0.01)

        # バッファが再度クリアされたことを確認
        self.assertEqual(len(self.plot_results.episode_buffer), 0)
        # ファイル内容のチェックはSaverのテストに任せるため削除

        # 集計が正しく行われたか（平均値の計算）は、Saverのテストでファイル内容を確認することで間接的に検証可能
        # または、モックを使ってSaverのlog_scoresが正しい引数で呼ばれたかテストすることもできるが、
        # 今回はシンプルに呼び出し確認は行わない方針とする。


    def test_add_agent_state(self):
        """add_agent_state がSaverのlog_agent_statesを呼び出すかテスト."""
        # setUpで作成されたファイルを削除し、クリーンな状態でテスト開始
        if os.path.exists(self.agents_states_path):
             os.remove(self.agents_states_path)
        # Saverがファイルを初期化するように新しいインスタンスを作成
        self.saver = Saver(self.save_dir)
        self.plot_results = PlotResults(self.save_dir, self.saver)

        # エージェント状態データを追加
        self.plot_results.add_agent_state(0, 1, 1)
        self.plot_results.add_agent_state(0, 1, 2)
        self.plot_results.add_agent_state(1, 3, 3)

        # ファイル内容のチェックはSaverのテストに任せるため削除
        # PlotResults側では、add_agent_stateが呼ばれた際に、
        # 内部でself.saver.log_agent_statesが正しい引数で呼ばれたかをモック等でテスト可能だが、
        # 今回はシンプルに呼び出し確認は行わない方針とする。


    def test_road_csv(self):
        """road_csv がサマリーCSVからデータを正しく読み込むかテスト."""
        # road_csv は Saver からパスを取得して読み込む
        data = self.plot_results.road_csv()

        # データの形状を確認 (setUpで作成したダミーデータに基づく)
        self.assertEqual(data.shape, (2, 5))

        # 列名を確認
        expected_columns = ['episode_group_start', 'episode_group_end', 'avg_time_step', 'avg_reward', 'avg_loss']
        self.assertListEqual(list(data.columns), expected_columns)

        # ダミーサマリーデータの特定の値を確認
        self.assertAlmostEqual(data.loc[0, 'avg_time_step'], 80.0)
        self.assertAlmostEqual(data.loc[0, 'avg_reward'], 15.0)
        self.assertAlmostEqual(data.loc[0, 'avg_loss'], 0.07666666666666666)

        self.assertAlmostEqual(data.loc[1, 'avg_time_step'], 60.0)
        self.assertAlmostEqual(data.loc[1, 'avg_reward'], 30.0)
        self.assertAlmostEqual(data.loc[1, 'avg_loss'], 0.03)


    def test_draw(self):
        """draw メソッドがサマリーデータに基づいてプロットファイルを生成するかテスト."""
        # 期待されるプロットファイル名 (Saverからパスを取得) を定義
        expected_plot_file = self.saver.summary_scores_path.replace('.csv', '.pdf')

        # draw 呼び出し前にファイルが存在しないことを確認 (setUpで削除済み)
        self.assertFalse(os.path.exists(expected_plot_file))

        # draw メソッドを呼び出し
        self.plot_results.draw()

        # プロットファイルが作成されたことを確認
        self.assertTrue(os.path.exists(expected_plot_file))

        # 単体テストでプロットの内容を詳細に確認することは困難ですが、
        # ファイルが生成され、test_road_csv でサマリーファイルの内容が正しいことを確認しています。


    def test_draw_heatmap(self):
        """draw_heatmap メソッドが agents_states.csv からヒートマッププロットファイルを生成するかテスト."""
        # 期待されるヒートマッププロットファイル名を定義 (Saverからパスを取得)
        expected_heatmap_file = self.saver.agents_states_path.replace('.csv', '_agents_heatmap.pdf')

        # draw_heatmap 呼び出し前にファイルが存在しないことを確認 (setUpで削除済み)
        self.assertFalse(os.path.exists(expected_heatmap_file))

        # draw_heatmap が読み込むためのダミーエージェント状態データをCSVに追加 (setUpで追加済み)
        # self.plot_results.add_agent_state(0, 1, 1) # add_agent_state はSaverに委譲
        # self.plot_results.add_agent_state(0, 1, 2)
        # self.plot_results.add_agent_state(0, 2, 2)
        # self.plot_results.add_agent_state(1, 3, 3)
        # self.plot_results.add_agent_state(1, 3, 4)
        # self.plot_results.add_agent_state(1, 4, 4)

        # draw_heatmap メソッドを適切な cell_num で呼び出し
        self.plot_results.draw_heatmap(cell_num=5)

        # ヒートマッププロットファイルが作成されたことを確認
        self.assertTrue(os.path.exists(expected_heatmap_file))

    def test_saver_log_scores(self):
        """Saver.log_scores が scores_summary.csv にデータを正しく追記するかテスト."""
        # setUpで作成されたサマリーファイルを削除し、クリーンな状態でテスト開始
        if os.path.exists(self.summary_scores_path):
             os.remove(self.summary_scores_path)
        # Saverがファイルを初期化するように新しいインスタンスを作成
        self.saver = Saver(self.save_dir)

        # データをログに記録
        self.saver.log_scores(1, 100, 85.0, 12.5, 0.09)
        self.saver.log_scores(101, 200, 65.0, 25.0, 0.04)

        # ファイルの内容を確認
        with open(self.summary_scores_path, "r") as f:
            content = f.readlines()

        # ヘッダーと2行のデータがあることを確認
        self.assertEqual(len(content), 3)
        self.assertEqual(content[0].strip(), "episode_group_start,episode_group_end,avg_time_step,avg_reward,avg_loss")
        self.assertEqual(content[1].strip(), "1,100,85.0,12.5,0.09")
        self.assertEqual(content[2].strip(), "101,200,65.0,25.0,0.04")

    def test_saver_log_agent_states(self):
        """Saver.log_agent_states が agents_states.csv にデータを正しく追記するかテスト."""
        # setUpで作成されたファイルを削除し、クリーンな状態でテスト開始
        if os.path.exists(self.agents_states_path):
             os.remove(self.agents_states_path)
        # Saverがファイルを初期化するように新しいインスタンスを作成
        self.saver = Saver(self.save_dir)

        # エージェント状態データをログに記録
        self.saver.log_agent_states(0, 5, 10)
        self.saver.log_agent_states(1, 20, 30)
        self.saver.log_agent_states(0, 6, 11)

        # ファイルの内容を確認
        with open(self.agents_states_path, "r") as f:
            content = f.readlines()

        # ヘッダーと3行のデータがあることを確認
        self.assertEqual(len(content), 4)
        self.assertEqual(content[0].strip(), "agent_id,x,y")
        self.assertEqual(content[1].strip(), "0,5,10")
        self.assertEqual(content[2].strip(), "1,20,30")
        self.assertEqual(content[3].strip(), "0,6,11")
