import os

from Base.IO_Base import BaseModelIO
from .QTable import QTableType

import os
import pickle

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

    def save(self, agent_id: int, q_table: QTableType) -> None:
        """
        指定されたエージェントIDのQテーブルデータをファイルに保存する (pickle形式).

        Args:
            agent_id (int): 保存するQテーブルのエージェントID.
            q_table (QTableType): 保存するQテーブルのデータ.
        """
        # file_path 引数は削除し、内部で self.model_dir を使用してパスを生成
        file_path = os.path.join(self.model_dir, f'agent_{agent_id}_Qtable.pkl')

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(q_table, f)
            print(f"Qテーブルを {file_path} に保存しました.")
        except Exception as e:
            print(f"Qテーブルの保存中にエラーが発生しました: {e}")
    
    def load(self, agent_id: int) -> QTableType:
        """
        指定されたエージェントIDのQテーブルデータをファイルから読み込む (pickle形式).

        Args:
            agent_id (int): 読み込むQテーブルのエージェントID.

        Returns:
            QTableType: 読み込まれたQテーブルのデータ. ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す.
        """
        # save_dir 引数は削除し、内部で self.model_dir を使用してパスを生成
        file_path = os.path.join(self.model_dir, f'agent_{agent_id}_Qtable.pkl')

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    q_table:QTableType = pickle.load(f)
                print(f"Qテーブルを {file_path} から読み込みました.")
            else:
                print(f"指定されたファイル {file_path} が存在しません。新しいQテーブルを作成します。")
                q_table:QTableType = {} # ファイルが存在しない場合は新しいQテーブルを初期化
        except Exception as e:
            print(f"Qテーブルの読み込み中にエラーが発生しました: {e}")
            q_table = {} # エラー発生時は新しいQテーブルを初期化

        return q_table
