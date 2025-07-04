"""

Saverクラスの設計仕様

1. Saverクラスの主な目的:
   学習プロセス中に生成される重要なデータ（モデルパラメータ、学習ログ、エージェントの状態など）を
   永続化（ファイルに保存）すること．これにより、学習の中断・再開や、学習結果の分析を可能にする．

2. Saverクラスが担当する具体的なタスク:
   - モデルパラメータの保存: 各エージェントの学習済みモデルパラメータ（例: Q値テーブルやニューラルネットワークの重み）をファイルに保存する．
     IQLの場合はエージェントごとに、CQLの場合は共通のパラメータを保存する必要がある．
     ファイル形式はCSVなどを想定．
   - 学習ログの保存: 各エピソードにおける報酬、ステップ数、損失などの学習の進捗に関する情報をファイルに保存する．
     通常はCSV形式で、エピソードごとに追記していく形式を想定．
   - エージェントの状態履歴の保存（オプション）: 学習中にエージェントが到達した状態の履歴を保存する．
     これはヒートマップ描画などの分析に利用できる．CSV形式を想定．
   - 保存先のディレクトリ構造の管理: 学習設定（mask, reward_mode, 環境サイズなど）に基づいた適切なディレクトリパスを生成し、
     必要であればディレクトリを作成する．
   - ファイルパスの生成: 保存するファイル（モデルファイル、ログファイル、状態ファイルなど）のパスを生成する．

3. Saverクラスが担当しないタスク:
   - 環境との相互作用: 環境のリセットやステップ実行などの環境シミュレーションは行わない．
   - エージェントの行動選択や学習処理: エージェントの行動決定ロジックや、Q値/ネットワークの更新などの学習アルゴリズムは担当しない．
   - 学習のメインループ制御: エピソードの実行やステップの進行を制御するメインループは担当しない．
   - 学習結果のプロットや可視化: 保存されたデータを読み込んでグラフを作成する処理は担当しない（これはPlotResultsのような別のクラスが担当）．

4. Saverクラスが他のクラスとどのように連携するか:
   - MultiAgent_Qクラス:
     - MultiAgent_Qは学習のメインループを管理しており、エピソードの終了時や一定間隔でSaverクラスのメソッドを呼び出し、
       モデルパラメータや学習ログの保存を指示する．
     - MultiAgent_Qは学習設定（mask, reward_modeなど）やファイルパスに関する情報をSaverクラスに渡す．
     - Saverクラスは、MultiAgent_Qクラスが持つエージェントインスタンスや学習に関するデータ（報酬、損失、ステップ数など）を
       受け取り、ファイルに書き込む処理を行う．
   - Agent_Qクラス:
     - Saverクラスがモデルパラメータを保存する際、Agent_Qクラスが保持する学習パラメータ（例: Qテーブルやネットワークパラメータ）に
       アクセスする必要がある．Agent_QクラスはこれらのパラメータをSaverクラスからアクセス可能な形で提供する必要がある（例えば、
       パラメータを取得するメソッドをAgent_Qに実装する）．
     - Agent_Qは自身の学習に関する情報（例: 損失）をMultiAgent_Qに渡し、MultiAgent_QがSaverにログ保存を依頼する．
   - GridWorldクラス:
     - 直接的な連携はない．SaverクラスはGridWorldの状態（エージェントやゴールの位置）を保存することがあるが、
       それはMultiAgent_Q経由で受け取った状態データを利用する形になる．

"""

import os
import numpy as np
import csv

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

class Saver:
    def __init__(self,save_dir,scores_path,agents_states_path):
        self.save_dir = save_dir
        self.scores_path = scores_path
        self.agents_states_path = agents_states_path

        self.mask = 0

        # なければ作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # スコアファイルを初期化（ヘッダ書き込み）
        with open(self.scores_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'time_step', 'reward', 'loss'])

        # ヘッダ書き込み
        with open(self.agents_states_path, 'w', newline='') as f:
            csv.writer(f).writerow(['episode', 'time_step', 'agent_id', 'agent_state'])

    def log_scores(self, episode, time_step, reward, loss):

        # 内容
        with open(self.scores_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, reward, loss])

    def log_agent_states(self, episode, time_step, agent_id, agent_state):

        if isinstance(agent_state, (list, tuple, np.ndarray)):
            state_str = '_'.join(map(str, agent_state))
        else:
            state_str = str(agent_state)
        with open(self.agents_states_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, agent_id, state_str])

    def save_model(self, agents):

        model_dir_path = os.path.join("output", self.save_dir,'model_weights')
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)

        print('パラメータ保存中...')
        
        for i, agent in enumerate(agents):
            path = (os.path.join(model_dir_path, f'{i}.csv')
                    if self.mask else
                    os.path.join(model_dir_path, 'common.csv'))
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Agent_Qクラスが持つ学習パラメータを取得する必要がある
                # 例: agent.get_model_params() のようなメソッドをAgent_Qに追加する必要があるかもしれない
                # 現在のコードではagent.linear.theta_listやagent.linear.common_theta_listを参照しているが、
                # Agent_Qクラスの内部実装に依存しすぎている
                # Agent_Qクラスにパラメータを公開するメソッドを追加するか、
                # ここで直接アクセス可能な構造になっているか確認が必要
                try:
                    if self.mask:
                        # IQLの場合、各エージェントが独自のthetaを持つと仮定
                        # Agent_Qクラスに self.theta_list を持たせる必要がある
                        data = agent.theta_list
                    else:
                        # CQLの場合、共通のthetaを持つと仮定
                        # Agent_Qクラスに self.common_theta_list を持たせる必要がある
                        # そして、それを2次元にリシェイプして保存
                        arr = np.array(agent.common_theta_list)
                        data = arr.reshape(-1, arr.shape[2]) if arr.ndim > 2 else arr
                except AttributeError as e:
                     print(f"エラー: Agent_Qクラスに学習パラメータを保持する変数がないか、名前が異なります: {e}")
                     print("Agent_Qクラスの実装を確認し、学習パラメータが self.theta_list または self.common_theta_list として保持されているか確認してください。")
                     return # 保存処理を中断

                for row in data:
                    writer.writerow(row)
        print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")
    
    # モデルパラメータの保存

    # 学習ログの保存
    def save_learn_histry(self):
        pass