import os
import torch


class IOHandler:
    """
    モデルデータやチェックポイントデータなどの保存と読み込みを担当する汎用クラス。
    torch.save/load を使用してデータをファイルに保存・読み込みする。
    """
    def save(self, save_container: dict, file_path:str) -> None:
        """
        指定されたデータをファイルに保存する (torch形式)。

        Args:
            file_path (str): 保存するファイルのパス。
            save_container (dict): チェックポイントに保存するデータ。
        """
        try:
            with open(file_path, 'wb') as f:
                torch.save(save_container, f)
            print(f"データを {file_path} に保存しました。")
        except Exception as e:
            print(f"データの保存中にエラーが発生しました: {e}")


    def load(self, file_path:str) -> dict:
        """
        指定されたファイルからデータを読み込む (torch形式)。

        Args:
            file_path (str): 読み込むファイルのパス。

        Returns:
            dict : 読み込まれたデータ。ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す。
        """

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    obj:dict = torch.load(f)
                print(f"データを {file_path} から読み込みました.")
            else:
                print(f"指定されたファイル {file_path} が存在しません。新しいデータとして初期化します。")
                obj = {} # ファイルが存在しない場合は新しいデータとして初期化
        except Exception as e:
            print(f"データの読み込み中にエラーが発生しました: {e}")
            obj = {} # エラー発生時は新しいデータを初期化

        return obj