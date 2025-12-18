import os

from Base.IO_Base import BaseModelIO
# from DQN.dqn import QNet

import os
import torch

class Model_IO(BaseModelIO):
    """
    モデルデータ（この場合はQテーブル）のファイルへの保存とファイルからの読み込みを担当するクラス.
    具体的なモデルの実装には依存せず、汎用的なIO処理を提供する.
    """

    def save(self, file_path, qnet_dict):
        # qnet_dict = qnet.state_dict()
        #file_path = os.path.join(self.model_dir, f"model_{agent_id}.pth")
        
        torch.save(qnet_dict, file_path)
        print(f"モデルを{file_path}に保存。")

    # qnetを渡す代わりに、既に用意したdictを受け取るようにすると汎用性が高まります
    def save_checkpoint(self, file_path, q_dict, t_dict, optim_dict, epoch, epsilon):
        checkpoint = {
            'model_state': q_dict,
            'target_state': t_dict, # 追加
            'optimizer_state': optim_dict,
            'epoch': epoch,
            'epsilon': epsilon
        }
        torch.save(checkpoint, file_path)
        # print(f"チェックポイントを上書き: {file_path} (Epoch: {epoch})")

    def load(self, file_path) -> dict:
        try:
            # torch.loadが返すのは QNetインスタンスではなく辞書(state_dict)
            load_data: dict = torch.load(file_path, weights_only=True) 
            print("モデルの重みをロード")
        except Exception as e:
            raise ValueError(f"エラー発生: {e}")
        
        return load_data

    def load_checkpoint(self, file_path) -> tuple[dict, dict, dict, int, float]:
        load_data = torch.load(file_path, weights_only=False)
        return (
            load_data['model_state'],
            load_data.get('target_state', load_data['model_state']), # 互換性のために無い場合はmodel_stateを流用
            load_data['optimizer_state'],
            load_data['epoch'],
            load_data['epsilon']
        )
