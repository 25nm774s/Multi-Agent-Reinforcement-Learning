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

    def save(self, file_path, qnet):
        qnet_dict = qnet.state_dict()
        #file_path = os.path.join(self.model_dir, f"model_{agent_id}.pth")
        
        torch.save(qnet_dict, file_path)
        print("saved model")

    def load(self, file_path) -> QNet:
        try:
            load_data: QNet = torch.load(file_path)
            print("model loaded")
        except Exception as e:
            raise ValueError(f"エラー発生: {e}")
        
        return load_data