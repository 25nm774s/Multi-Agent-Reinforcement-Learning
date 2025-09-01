import os

from Base.IO_Base import BaseModelIO, BaseCheckpointHandlar
from .QTable import QTableType

import os
import torch
#import pickle

class CheckPointHandler(BaseCheckpointHandlar):
    """
    モデルデータ（この場合はQテーブル）のファイルへの保存とファイルからの読み込みを担当するクラス.
    具体的なモデルの実装には依存せず、汎用的なIO処理を提供する.
    """
    def save(self, obj: dict, file_path:str) -> None:
        """
        指定されたエージェントIDのQテーブルデータをファイルに保存する (pickle形式).

        Args:
            agent_id (int): 保存するQテーブルのエージェントID.
            q_table (QTableType): 保存するQテーブルのデータ.
        """
        #file_path = os.path.join(model_dir, f'agent_{agent_id}_Qtable.pkl')

        try:
            with open(file_path, 'wb') as f:
                torch.save(obj, f)
            print(f"モデル情報を {file_path} に保存しました.")
        except Exception as e:
            print(f"モデル情報の保存中にエラーが発生しました: {e}")

    
    def load(self, file_path:str) -> dict:
        """
        指定されたエージェントIDのQテーブルデータをファイルから読み込む (pickle形式).

        Args:
            file_path (str): 読み込むQテーブルのエージェントID.

        Returns:
            QTableType: 読み込まれたQテーブルのデータ. ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す.
        """

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    obj:dict = torch.load(f)
                print(f"Qテーブルを {file_path} から読み込みました.")
            else:
                raise OSError(f"指定されたファイル {file_path} が存在しません。")
                #q_table:QTableType = {} # ファイルが存在しない場合は新しいQテーブルを初期化
        except Exception as e:
            print(f"Qテーブルの読み込み中にエラーが発生しました: {e}")
            obj = {} # エラー発生時は新しいQテーブルを初期化

        return obj

class Model_IO(BaseModelIO):
    def save(self, qtable: QTableType, file_path:str) -> None:
        """
        指定されたエージェントIDのQテーブルデータをファイルに保存する (pickle形式).

        Args:
            q_table (QTableType): 保存するQテーブルのデータ.
            file_path (str): 保存するQテーブルのパス.
        """
        #file_path = os.path.join(model_dir, f'agent_{agent_id}_Qtable.pkl')

        try:
            with open(file_path, 'wb') as f:
                #pickle.dump(qtable, f)
                torch.save(qtable, f)
            print(f"Qテーブルを {file_path} に保存しました.")
        except Exception as e:
            print(f"Qテーブルの保存中にエラーが発生しました: {e}")

    
    def load(self, file_path:str) -> dict:
        """
        指定されたエージェントIDのQテーブルデータをファイルから読み込む (pickle形式).

        Args:
            file_path (str): 読み込むQテーブルのエージェントID.

        Returns:
            QTableType: 読み込まれたQテーブルのデータ. ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す.
        """

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    q_table:QTableType = torch.load(f)
                print(f"Qテーブルを {file_path} から読み込みました.")
            else:
                print(f"指定されたファイル {file_path} が存在しません。新しいQテーブルを初期化します。")
                q_table:QTableType = {} # ファイルが存在しない場合は新しいQテーブルを初期化
        except Exception as e:
            print(f"Qテーブルの読み込み中にエラーが発生しました: {e}")
            q_table = {} # エラー発生時は新しいQテーブルを初期化

        return q_table
