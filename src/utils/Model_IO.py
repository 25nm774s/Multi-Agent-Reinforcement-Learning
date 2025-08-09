import os
import pickle

from Q_learn.QTable import QTableType

class Model_IO:
    def __init__(self, model_dir_path):
        self.model_dir_path = model_dir_path
        os.makedirs(self.model_dir_path, exist_ok=True)

    def save_q_table(self, agent_id, q_table_data:QTableType) -> None:
        """
        Qテーブルをファイルに保存する (pickle形式).
        """
        file_path = os.path.join(self.model_dir_path, f'agent_{agent_id}_q_table.pkl')
        # ディレクトリ部分を取得し、空でない場合にディレクトリを作成
        save_dir = os.path.dirname(file_path)
        if save_dir: # ディレクトリが指定されている場合のみ作成
            os.makedirs(save_dir, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(q_table_data, f)
            print(f"Qテーブルを {file_path} に保存しました.")
        except Exception as e:
            print(f"Qテーブルの保存中にエラーが発生しました: {e}")

    def load_q_table(self, agent_id) -> QTableType:
        """
        ファイルからQテーブルを読み込む (pickle形式).
        """
        file_path = os.path.join(self.model_dir_path, f'agent_{agent_id}_q_table.pkl')
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    q_table_data:QTableType = pickle.load(f)
                print(f"Qテーブルを {file_path} から読み込みました.")
                return q_table_data
            else:
                print(f"指定されたファイル {file_path} が存在しません。新しいQテーブルを作成します。")
                return {} # ファイルが存在しない場合は新しいQテーブルを初期化
        except Exception as e:
            print(f"Qテーブルの読み込み中にエラーが発生しました: {e}")
            return {} # エラー発生時は新しいQテーブルを初期化
