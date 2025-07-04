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

class Saver(ISaver):
	"""
    ファイルに学習結果を保存する Saver クラスの実装。
    BaseSaver を継承します。
    """
	def __init__(self, save_dir, mask):
		super().__init__(save_dir)
		self.mask = mask # mask の情報を Saver で持つように変更

		self.scores_path = os.path.join(self.save_dir, "scores.csv")
		self.agents_states_path = os.path.join(self.save_dir, "agents_states.csv")

		# スコアファイルを初期化（ヘッダ書き込み）
		with open(self.scores_path, 'w', newline='') as f:
			csv.writer(f).writerow(['episode', 'time_step', 'reward', 'loss'])

		# エージェント状態ファイルを初期化（ヘッダ書き込み）
		with open(self.agents_states_path, 'w', newline='') as f:
			csv.writer(f).writerow(['episode', 'time_step', 'agent_id', 'agent_state'])


	def log_scores(self, episode, time_step, reward, loss):
		"""
		学習スコアをCSVファイルに記録します。
		"""
		with open(self.scores_path, 'a', newline='') as f:
			csv.writer(f).writerow([episode, time_step, reward, loss])

	def log_agent_states(self, episode, time_step, agent_id, agent_state):
		"""
		エージェントの状態をCSVファイルに記録します。
		"""
		# 状態がリストやタプル、NumPy配列の場合は文字列に変換
		if isinstance(agent_state, (list, tuple, np.ndarray)):
			state_str = '_'.join(map(str, agent_state))
		else:
			state_str = str(agent_state)

		with open(self.agents_states_path, 'a', newline='') as f:
			csv.writer(f).writerow([episode, time_step, agent_id, state_str])

	def save_model(self, agents):
		"""
		学習済みモデルのパラメータをCSVファイルに保存します。
		"""
		model_dir_path = os.path.join(self.save_dir, 'model_weights')
		if not os.path.exists(model_dir_path):
			os.makedirs(model_dir_path)

		print('パラメータ保存中...')

		for i, agent in enumerate(agents):
			# mask に応じてファイルパスを決定
			path = (os.path.join(model_dir_path, f'{i}.csv')
					if self.mask else
					os.path.join(model_dir_path, 'common.csv'))

			with open(path, 'w', newline='') as f:
				writer = csv.writer(f)
				try:
					# Agent_Qクラスにパラメータ取得用のメソッドがあると理想的ですが、
					# 現在のコードに合わせて直接アクセスを試みます。
					# Agent_Qクラスの実装に依存するため、注意が必要です。
					if self.mask:
						# IQLの場合
						data = agent.theta_list # Agent_Qに theta_list がある前提
					else:
						# CQLの場合
						arr = np.array(agent.common_theta_list) # Agent_Qに common_theta_list がある前提
						data = arr.reshape(-1, arr.shape[2]) if arr.ndim > 2 else arr

				except AttributeError as e:
						print(f"エラー: Agent_Qクラスに学習パラメータを保持する変数がないか、名前が異なります: {e}")
						print("Agent_Qクラスの実装を確認し、学習パラメータが self.theta_list または self.common_theta_list として保持されているか確認してください。")
						return # 保存処理を中断

				for row in data:
					writer.writerow(row)
		print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")

	def save_model_weights(self,agents):
		model_dir_path = os.path.join("output", self.save_dir,'model_weights')
		if not os.path.exists(model_dir_path):
			os.makedirs(model_dir_path)

		print('モデル保存中...')
		for i, agent in enumerate(agents):
			torch.save(agent.model.qnet.state_dict(), self.model_path[i])
		print(f"保存先: {GREEN}{model_dir_path}{RESET}\n")

    # 必要に応じて load_model などのメソッドを実装
    # def load_model(self, agent_id):
    #     pass