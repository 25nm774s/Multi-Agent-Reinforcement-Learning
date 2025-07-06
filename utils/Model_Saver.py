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
import torch

#from BaseAgent import Agent#←のように抽象クラスをimportするとよい

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

# 基底クラス
from abc import ABC, abstractmethod

class ISaver(ABC):
    """
    Saver クラスの抽象基底クラス。
    データ永続化に関する共通インターフェースを定義します。
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        # 保存ディレクトリがなければ作成
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @abstractmethod
    def log_scores(self, episode, time_step, reward, loss):
        """
        学習スコアを記録する抽象メソッド。
        派生クラスで実装する必要があります。
        """
        pass

    @abstractmethod
    def log_agent_states(self, episode, time_step, agent_id, agent_state):
        """
        エージェントの状態を記録する抽象メソッド。
        派生クラスで実装する必要があります。
        """
        pass

    @abstractmethod
    def save_model(self, agents):
        """
        学習済みモデルのパラメータを保存する抽象メソッド。
        派生クラスで実装する必要があります。
        """
        pass

    @abstractmethod
    def save_model_weights(self):
        """
        学習済みモデルの重みを保存する抽象メソッド。
        """
        pass

    ## 必要に応じて他の共通メソッドや抽象メソッドを追加できます
    ## 例:
    ## @abstractmethod
    ## def load_model(self, agent_id):
    ##     pass

import os
import csv
import torch
import numpy as np

class Saver:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.scores_path = os.path.join(self.save_dir, "scores.csv")
        self.agents_states_path = os.path.join(self.save_dir, "agents_states.csv")
        self.model_dir_path = os.path.join(self.save_dir, 'model_weights')

        # Ensure directories exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir_path, exist_ok=True)

        # Initialize log files with headers if they don't exist
        if not os.path.exists(self.scores_path):
            with open(self.scores_path, 'w', newline='') as f:
                csv.writer(f).writerow(['episode', 'time_step', 'reward', 'loss'])

        if not os.path.exists(self.agents_states_path):
             with open(self.agents_states_path, 'w', newline='') as f:
                csv.writer(f).writerow(['episode', 'time_step', 'agent_id', 'agent_state'])


    def save_q_table(self, agents, mask):
        print('Qテーブル保存中...')
        for i, agent in enumerate(agents):
            path = (os.path.join(self.model_dir_path, f'{i}.csv')
                    if mask else
                    os.path.join(self.model_dir_path, 'common.csv'))

            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                if mask:
                    # Assuming Agent_Q stores Q table in agent.linear.theta_list
                    data = agent.linear.theta_list
                else:
                    # Assuming common Q table is in agent.linear.common_theta_list
                    # and needs reshaping
                    arr = np.array(agent.linear.common_theta_list)
                    # Reshape to 2D: (states * actions) x 1
                    data = arr.reshape(-1, arr.shape[2])
                for row in data:
                    writer.writerow(row)
            if not mask: # Only save the common table once
                break
        print(f"保存先: {self.model_dir_path}\n")


    def save_dqn_weights(self, agents):
        print('モデル重み保存中...')
        for i, agent in enumerate(agents):
            path = os.path.join(self.model_dir_path, f"{i}.pth")
            # Assuming Agent_DQN stores the model in agent.model.qnet
            torch.save(agent.model.qnet.state_dict(), path)
        print(f"保存先: {self.model_dir_path}\n")

    def log_agent_states(self, episode, time_step, agent_id, agent_state):
        if isinstance(agent_state, (list, tuple, np.ndarray)):
            state_str = '_'.join(map(str, agent_state))
        else:
            state_str = str(agent_state)
        with open(self.agents_states_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, agent_id, state_str])

    def log_scores(self, episode, time_step, reward, loss):
        with open(self.scores_path, 'a', newline='') as f:
            csv.writer(f).writerow([episode, time_step, reward, loss])
