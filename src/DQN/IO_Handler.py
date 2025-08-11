import os

from Base.IO_Base import BaseModelIO
from DQN.dqn import QNet

import os
import torch

class Model_IO(BaseModelIO):
    """
    モデルデータ（この場合はQテーブル）のファイルへの保存とファイルからの読み込みを担当するクラス.
    具体的なモデルの実装には依存せず、汎用的なIO処理を提供する.
    """
    def __init__(self, save_dir):
        """
        Model_IOクラスのコンストラクタ.

        Args:
            save_dir (str): モデルを保存するルートディレクトリのパス.
        """
        self.model_dir: str = os.path.join(save_dir, "models")

        # モデル保存ディレクトリが存在しない場合は作成
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print(f"ディレクトリ {self.model_dir} を作成しました。")

    def save(self, agent_id: int, qnet: QNet):
        qnet_dict = qnet.state_dict()
        file_path = os.path.join(self.model_dir, f"model_{agent_id}.pth")
        
        torch.save(qnet_dict, file_path)
        print("saved model")

    def load(self, agent_id) -> QNet:
        file_path: str  = os.path.join(self.model_dir, f"model_{agent_id}.pth")
        try:
            load_data: QNet = torch.load(file_path)
            print("model loaded")
        except Exception as e:
            raise ValueError(f"エラー発生: {e}")
        
        return load_data