import unittest
import os
import json
import hashlib # ConfigManager クラスの定義に必要ですが、テストからも参照します
import shutil # フォルダ削除のために shutil をインポート

# Assume ConfigLoader, HashManager, ConfigManager classes are defined in previous cells
from src.utils.ConfigManager import ConfigManager

class TestConfigManager(unittest.TestCase):

    def setUp(self):
        """各テストメソッドの実行前に呼び出される準備メソッド"""
        #self.test_dir = "test_config_manager_folder" # テスト用フォルダ名
        self.test_dir = "folder" # テスト用フォルダ名
        # テスト用フォルダを作成（既に存在する場合は何もしない）
        self.test_file_path = os.path.join(self.test_dir,"config_j.json")
        self.test_hash_file_path = os.path.join(self.test_dir, "config_h.hash")
        # テストファイルを削除してクリーンな状態にする
        #self._cleanup_files()
        self._cleanup_folder() # フォルダごと削除

    def tearDown(self):
        """各テストメソッドの実行後に呼び出されるクリーンアップメソッド"""
        # テスト用フォルダごと削除してクリーンアップする
        #self._cleanup_folder()
        pass

    def _cleanup_files(self):
        """テストファイルを削除するヘルパーメソッド"""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
            #print(f"Cleaned up {self.test_file_path}")
        if os.path.exists(self.test_hash_file_path):
            os.remove(self.test_hash_file_path)
            #print(f"Cleaned up {self.test_hash_file_path}")

    def _cleanup_folder(self):
        """テスト用フォルダを削除するヘルパーメソッド"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            #print(f"Cleaned up folder {self.test_dir}")


    def _create_test_file(self, data=None, include_hash=True, tampered_content=False):
        """テスト用の設定ファイルとハッシュファイルを作成するヘルパーメソッド"""
        os.makedirs(self.test_dir, exist_ok=True)
        if data is None:
            data = {
                "data1-str": "Initial Value",
                "data2-float": 2.5,
                "data3-int": 50,
                "data4-bool": True
            }

        file_content = json.dumps(data, indent=4)

        if tampered_content:
            # 内容を意図的に変更してハッシュ不一致を引き起こす
            tampered_data = data.copy()
            tampered_data["data3-int"] = data.get("data3-int", 0) + 1 # 値を少し変える
            file_content = json.dumps(tampered_data, indent=4)


        # 設定ファイルを作成
        with open(self.test_file_path, "w") as f:
            f.write(file_content)

        # ハッシュファイルを作成（include_hash が True の場合のみ）
        if include_hash:
            # ファイル内容のハッシュを計算
            current_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
            with open(self.test_hash_file_path, "w") as f:
                f.write(current_hash)


    # --- テストケース ---
    def test_01_new_creation(self):
        """新規作成シナリオのテスト"""
        # ファイルが存在しないことを確認
        self.assertFalse(os.path.exists(self.test_file_path))
        self.assertFalse(os.path.exists(self.test_hash_file_path))

        # ConfigManagerを初期化（新規作成されるはず）
        # config = ConfigManager(self.test_file_path)
        conf = ConfigManager({"data1-str": "Initial Value", "qqq": 333},self.test_dir)

        # 設定ファイルとハッシュファイルが作成されたことを確認
        self.assertTrue(os.path.exists(self.test_file_path))
        self.assertTrue(os.path.exists(self.test_hash_file_path))

        # 読み込まれたデータがデフォルト値であることを確認
        self.assertEqual(conf.get_setting("data1-str"), "Initial Value")
        self.assertEqual(conf.get_setting("qqq"), 333)

        # 現在のハッシュと保存されたハッシュが一致することを確認
        self.assertIsNotNone(conf.get_current_file_hash())
        self.assertEqual(conf.get_current_file_hash(), conf.get_stored_hash())

        # ハッシュファイルの内容が正しいか確認
        with open(self.test_hash_file_path, "r") as f:
            stored_hash_in_file = f.read().strip()
        self.assertEqual(stored_hash_in_file, conf.get_current_file_hash())

    def test_02_normal_load(self):
        """正常読み込みシナリオのテスト"""
        # 事前に正常な設定ファイルとハッシュファイルを作成
        data = {"key": "Value"}
        self._create_test_file(data=data,include_hash=True)

        # ConfigManagerを初期化（正常に読み込まれるはず）
        conf = ConfigManager(initial_data=data,config_dir=self.test_dir)

        # 読み込まれたデータが事前のデータと一致することを確認
        self.assertEqual(conf.data, data)

        # 現在のハッシュと保存されたハッシュが一致することを確認
        self.assertIsNotNone(conf.get_current_file_hash())
        self.assertIsNotNone(conf.get_stored_hash())
        self.assertEqual(conf.get_current_file_hash(), conf.get_stored_hash())

    def test_03_missing_hash_file_error(self):
        """設定ファイルは存在するがハッシュファイルがない場合のエラーテスト"""
        # 事前に設定ファイルだけを作成し、ハッシュファイルは作成しない
        initial_data = {"key": "value"}
        #os.makedirs(self.test_dir, exist_ok=True)
        #with open(self.test_file_path, "w") as f:
        #    json.dump(initial_data, f, indent=4)

        self._create_test_file(data=initial_data, include_hash=False)

        # ハッシュファイルが存在しないことを確認
        self.assertTrue(os.path.exists(self.test_file_path))
        self.assertFalse(os.path.exists(self.test_hash_file_path))

        # ConfigManager初期化時にIOErrorが発生することを確認
        with self.assertRaisesRegex(IOError, "エラー: ハッシュファイル .* が見つかりません。"):
            ConfigManager(initial_data,config_dir=self.test_dir)


    def test_04_content_tampering_error(self):
        """設定ファイルの内容が変更されている場合のエラーテスト"""
        # 事前に正常な設定ファイルとハッシュファイルを作成
        initial_data = {"key": "value", "number": 100}
        self._create_test_file(data=initial_data, include_hash=True)

        # 設定ファイルの内容を意図的に変更（ハッシュは変更しない）
        tampered_data = initial_data.copy()
        tampered_data["number"] = 200 # 値を変更
        #os.makedirs(self.test_dir, exist_ok=True)
        with open(self.test_file_path, "w") as f:
            json.dump(tampered_data, f, indent=4)

        hash = hashlib.sha256(json.dumps(initial_data).encode('utf-8')).hexdigest()
        with open(self.test_hash_file_path, "w") as f:
            f.write(hash)

        # 設定ファイルとハッシュファイルが存在することを確認
        self.assertTrue(os.path.exists(self.test_file_path))
        self.assertTrue(os.path.exists(self.test_hash_file_path))

        # ConfigManager初期化時にValueErrorが発生することを確認
        with self.assertRaisesRegex(ValueError, "設定ファイル .* の内容が変更された可能性があります"):
            ConfigManager(initial_data)
    '''
    def test_05_invalid_json_error(self):
        """設定ファイルが不正なJSON形式の場合のエラーテスト"""
        # 事前に不正なJSON形式のファイルを作成
        invalid_json_content = '{"key": "value", "invalid"}' # JSON形式として不正
        os.makedirs(self.test_dir, exist_ok=True)
        with open(self.test_file_path, "w") as f:
            f.write(invalid_json_content)

        # ハッシュファイルは作成しない（JSONパースエラーが先に発生するはず）
        self.assertTrue(os.path.exists(self.test_file_path))
        self.assertFalse(os.path.exists(self.test_hash_file_path))


        # ConfigManager初期化時にJSONDecodeErrorが発生することを確認
        with self.assertRaises(json.JSONDecodeError):
            ConfigManager(invalid_json_content,config_dir=self.test_file_path)#type:ignore 理由: あえて不正な辞書を渡すため。
        '''

    def test_06_file_not_found_error(self):
        """設定ファイルが存在しない場合のエラーテスト"""
        # 設定ファイルが存在しないことを確認
        self.assertFalse(os.path.exists(self.test_file_path))
        self.assertFalse(os.path.exists(self.test_hash_file_path))

        self._create_test_file(data=None, include_hash=False)
        os.remove(self.test_file_path)

        # 設定ファイルが存在しないことを確認
        self.assertFalse(os.path.exists(self.test_file_path))
        # ConfigManager初期化時にFileNotFoundErrorが発生することを確認
        with self.assertRaises(FileNotFoundError):
            ConfigManager({"key": "value"})
