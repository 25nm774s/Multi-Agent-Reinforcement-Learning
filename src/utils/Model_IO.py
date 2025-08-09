import os

from agent import Agent
from Q_learn.QTable import QTableType

class Model_IO:
    def __init__(self, model_dir:str):
        """
        モデルの保存・ロードを担当

        args:
            model_dir(str):モデル保存のディレクトリ (ex)./output/Q_Reward[0]_...
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def save_q_table(self, agents:list[Agent]) -> None:
        """
        Qテーブルをファイルに保存する (pickle形式).
        """
        for agent in agents:
            agent.save_q_table(self.model_dir)

    def load_q_table(self, agents:list[Agent]) -> list[QTableType]:
        """
        ファイルからQテーブルを読み込む (pickle形式).
        """
        load_data:list[QTableType] = []
        for agent in agents:
            qtable:QTableType = agent.load_q_table(self.model_dir)
            load_data.append(qtable)
        
        return load_data