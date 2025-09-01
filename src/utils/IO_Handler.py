import os

from Base.IO_Base import BaseModelIO, BaseCheckpointHandlar

import os
import torch


class CheckPointHandler(BaseCheckpointHandlar):
    """
    モデルデータのチェックポイントファイルへの保存とチェックポイントからの読み込みを担当するクラス。
    具体的なモデルの実装には依存せず、汎用的なIO処理を提供する。
    """
    def save(self, save_container: dict, file_path:str) -> None:
        """
        指定されたエージェントIDのQテーブルデータをファイルに保存する (pickle形式)。

        Args:
            file_path (str): 保存するQテーブルのファイルパス。
            save_container (dict): チェックポイントに保存するデータ。
        """
        try:
            with open(file_path, 'wb') as f:
                torch.save(save_container, f)
            print(f"モデル情報を {file_path} に保存しました。")
        except Exception as e:
            print(f"モデル情報の保存中にエラーが発生しました: {e}")

    
    def load(self, file_path:str) -> dict:
        """
        指定されたエージェントIDのQテーブルデータをファイルから読み込む (pickle形式)。

        Args:
            file_path (str): 読み込むQテーブルのエージェントID。

        Returns:
            dict : 読み込まれたセーブデータ。ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す。
        """

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    obj:dict = torch.load(f)
                print(f"セーブデータを {file_path} から読み込みました.")
            else:
                print(f"指定されたファイル {file_path} が存在しません。")
                obj = {} # ファイルが存在しない場合は新しいセーブデータとして初期化
        except Exception as e:
            print(f"セーブデータの読み込み中にエラーが発生しました: {e}")
            obj = {} # エラー発生時は新しいセーブデータを初期化

        return obj

class Model_IO(BaseModelIO):
    def save(self, model, file_path:str) -> None:
        """
        指定されたエージェントIDのQテーブルデータをファイルに保存する (torch形式)。

        Args:
            model : 保存するQテーブルのデータ。
            file_path (str): 保存するQテーブルのパス。
        """
        try:
            with open(file_path, 'wb') as f:
                torch.save(model, f)
            print(f"モデルを {file_path} に保存しました。")
        except Exception as e:
            print(f"モデルの保存中にエラーが発生しました: {e}")

    
    def load(self, file_path:str) -> dict:
        """
        指定されたエージェントIDのQテーブルデータをファイルから読み込む (torch形式)。

        Args:
            file_path (str): 読み込むQテーブルのファイルパス。

        Returns:
            dict : 読み込まれたQテーブルのデータ。ファイルが存在しない場合や読み込みエラーが発生した場合は空の辞書を返す。
        """

        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    model = torch.load(f)
                print(f"モデル重みを {file_path} から読み込みました.")
            else:
                print(f"指定されたファイル {file_path} が存在しません。新しく重みを初期化します。")
                model = {} # ファイルが存在しない場合は新しいQテーブルを初期化
        except Exception as e:
            print(f"モデルの読み込み中にエラーが発生しました: {e}")
            model = {} # エラー発生時は新しいQテーブルを初期化

        return model
