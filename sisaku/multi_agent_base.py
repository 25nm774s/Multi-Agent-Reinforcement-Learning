# src/multi_agent_base.py

import os
import sys
from abc import ABC, abstractmethod # 抽象基底クラスと抽象メソッドのためにインポート

# 共通ユーティリティクラスのインポート（仮定）
# プロジェクト構造に合わせてパスを調整してください
try:
    from utils.saver import Saver
    from utils.plot_results import PlotResults
    # 環境クラスも共通で使うはずなのでインポート
    # from envs.grid_world import GridWorld # 環境クラスのパスも調整してください
except ImportError as e:
    print(f"ユーティリティまたは環境クラスのインポートに失敗しました: {e}")
    print("src/utils または 環境クラスのパスが正しいか確認してください。")
    # エラーハンドリング、または適切なモック/ダミー実装を検討

# カラーコード（必要であれば）
GREEN = '\033[92m'
RESET = '\033[0m'


class MultiAgentBase(ABC):
    """
    複数のエージェントを用いた強化学習の実行を管理する基底クラス.
    環境とのインタラクション、エピソードの進行、結果の保存・表示など、
    学習アルゴリズムに依存しない共通の処理を定義します。
    """
    def __init__(self, args, agents):
        """
        MultiAgentBase クラスのコンストラクタ.

        Args:
            args: 実行設定を含むオブジェクト.
            agents (list): 使用するエージェントオブジェクトのリスト (具体的な型は派生クラスによる).
        """
        # 環境の初期化 (GridWorld が共通で使える場合)
        # self.env = GridWorld(args) # 環境クラスのインスタンス化は派生クラスで行うか、
                                     # ここで共通の環境クラスをインスタンス化する
        # ここでは環境のインスタンスは派生クラスが持つものと仮定し、
        # 基底クラスは環境オブジェクトを直接は保持しない、あるいは引数で受け取る形も考えられる
        # シンプルに、環境のインスタンスは派生クラスで作成し、必要に応じてメソッドに渡す、または
        # 基底クラスの __init__ で共通の環境クラスをインスタンス化する形にする。
        # GridWorldが共通で使えると仮定してここでインスタンス化します。
        try:
             # 環境クラスのパスを適切に設定してください
             from envs.grid_world import GridWorld # 例: src/envs ディレクトリに GridWorld がある場合
             self.env = GridWorld(args)
        except ImportError as e:
             print(f"環境クラスのインポートに失敗しました: {e}")
             print("環境クラスのパスが正しいか確認してください。")
             self.env = None # またはエラーを発生させる

        self.agents = agents

        self.reward_mode = args.reward_mode
        self.render_mode = args.render_mode
        self.episode_num = args.episode_number
        self.max_ts = args.max_timestep
        self.agents_num = args.agents_number
        self.goals_num = args.goals_number
        self.grid_size = args.grid_size

        self.load_model = args.load_model
        self.mask = args.mask # Q学習/DQNでマスクの扱いが異なる可能性あり

        # 結果保存ディレクトリの設定と作成
        # ディレクトリ名はアルゴリズム固有の情報を含むべきなので、
        # 派生クラスで設定するか、argsからアルゴリズム名を渡すなどの工夫が必要
        # ここでは基底クラスで共通の設定部分のみを行う例を示す
        # save_dir = os.path.join("output", "common_base_dir") # 例
        # if not os.path.exists(save_dir): os.makedirs(save_dir)
        # self.saver = Saver(save_dir) # Saverのインスタンス化も派生クラスで行う方が柔軟かも
        # self.plot_results = PlotResults(save_dir) # 同上

        # アルゴリズム固有のディレクトリ名は派生クラスで設定し、
        # SaverとPlotResultsのインスタンス化も派生クラスで行う方が自然です。
        # 基底クラスではインスタンス変数だけ定義しておきます。
        self.saver = None
        self.plot_results = None

        # 事前条件チェック: ゴール数はエージェント数以下である必要がある
        if self.agents_num < self.goals_num:
            print('goals_num <= agents_num に設定してください.\n')
            sys.exit()


    @abstractmethod
    def run(self):
        """
        強化学習のメイン実行ループ.
        このメソッドは各派生クラスで具体的な学習アルゴリズムに基づいて実装されます。
        """
        pass # 抽象メソッドなので実装は不要

    @abstractmethod
    def save_results(self):
        """
        学習結果（モデル重みやQテーブルなど）を保存する抽象メソッド.
        具体的な保存内容は派生クラスで実装されます。
        """
        pass # 抽象メソッドなので実装は不要

    def result_show(self):
        """
        学習結果をプロットして表示する共通メソッド.
        saver および plot_results インスタンスは派生クラスで適切に初期化されていることを前提とします。
        """
        if self.plot_results is None:
            print("PlotResults インスタンスが初期化されていません。")
            return

        print("Showing results...")
        self.plot_results.draw()
        # draw_heatmap は環境サイズに依存するので、引数が必要
        self.plot_results.draw_heatmap(self.grid_size)


# 補足：Agent クラスも共通の基底クラスを持つと、MultiAgentBase の agents リストの型ヒントを
# より具体的に記述できるようになります (例: agents: list[AgentBase])。
# これは必須ではありませんが、設計の一貫性を高めます。