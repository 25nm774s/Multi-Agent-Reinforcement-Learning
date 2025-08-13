import hashlib
import os
import json

class ConfigManager:
    """
    設定ファイルの読み込み、ハッシュ値による整合性チェック、
    および設定データへのアクセスを提供するクラス。
    設定ファイルとハッシュファイルがペアで存在することを前提とし、
    不整合がある場合はエラーを発生させる。
    """
    def __init__(self, file_path="config.json"):
        """
        ConfigManagerの新しいインスタンスを初期化する。
        指定されたファイルパスから設定とハッシュを読み込み、整合性をチェックする。
        設定ファイルが存在しない場合は新規作成し、そのハッシュを保存する。

        Args:
            file_path (str): 設定ファイルのパス。対応するハッシュファイルは .hash 拡張子で管理される。

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合 (ConfigLoaderが処理)。
            json.JSONDecodeError: 設定ファイルのJSON形式が不正な場合 (ConfigLoaderが処理)。
            IOError: ファイルの読み書き中にエラーが発生した場合、または設定ファイルは存在するがハッシュファイルが見つからない場合。
            ValueError: 設定ファイルの内容と保存されたハッシュ値が一致しない場合。
            Exception: 上記以外の予期しないエラーが発生した場合。
        """
        self.file_path = file_path
        self.hash_file_path = self.file_path + ".hash"
        self.data = {}
        self._current_file_hash = None
        self._stored_hash = None

        # Instantiate helper classes
        self._config_loader = ConfigLoader(self.file_path)
        self._hash_manager = HashManager()

        # 設定ファイルが初期化開始時に存在したかどうかのフラグ
        config_file_existed_initially = os.path.exists(self.file_path)

        # Step 3: Load or create the configuration
        # This handles FileNotFoundError for the config file itself.
        # It also handles JSONDecodeError and IOError during loading.
        try:
            self._config_loader.load_or_create()
            self.data = self._config_loader.get_data()
        except Exception as e: # Catching broader exception to add context if needed
            print(f"エラー: 設定ファイル '{self.file_path}' の処理中にエラーが発生しました: {e}")
            raise # Re-raise the original exception


        # Get the content as string to calculate hash.
        try:
            with open(self.file_path, "r") as f:
                file_content = f.read()
        except IOError as e:
             # Should not happen if _config_loader.load_or_create() was successful.
             # If config file existed initially, ConfigLoader.load_or_create would have handled read errors.
             # If config file was newly created, ConfigLoader.load_or_create would have handled write errors.
             # This might catch rare timing issues or permission changes after load/create but before re-reading for hash.
             raise IOError(f"Error reading file content after load/create: {e}")


        # Step 4: Calculate current hash based on the content read
        self._current_file_hash = self._hash_manager.calculate_hash(file_content)

        # Step 5: Attempt to load the stored hash
        # This returns None if the hash file does not exist, or raises IOError on read error.
        try:
            self._stored_hash = self._hash_manager.load_hash(self.hash_file_path)
        except IOError as e:
            # HashManager.load_hash raised an IOError (e.g., permission error reading hash file)
            raise IOError(f"エラー: ハッシュファイル '{self.hash_file_path}' の読み込み中にエラーが発生しました: {e}")


        # Now handle integrity checks based on whether the config file existed initially

        if config_file_existed_initially:
            # Case: Config file existed when we started.
            # We MUST find a corresponding hash file based on the strict requirement.
            if self._stored_hash is None:
                 # Config file existed, but hash file was NOT found by load_hash.
                 # This is the scenario for the missing hash file error test.
                 # We need to explicitly check if the hash file still doesn't exist,
                 # as load_hash returning None only means it wasn't there *at that exact moment*.
                 # But since load_or_create succeeded on the config file, if _stored_hash is None,
                 # it means the hash file is missing.
                 raise IOError(
                    f"エラー: 設定ファイル '{self.file_path}' は存在しますが、"
                    f"対応するハッシュファイル '{self.hash_file_path}' が見つかりません。\n"
                    f"これは設定ファイルが不正に変更されたか、初回起動時のハッシュ保存に失敗した可能性があります。"
                )
            else:
                # Config file existed, hash file existed, compare hashes.
                if self._current_file_hash != self._stored_hash:
                    # Step 7: Raise ValueError on mismatch
                    raise ValueError(
                        f"エラー: 設定ファイル '{self.file_path}' の内容が変更された可能性があります！\n"
                        f"  現在のハッシュ: {self._current_file_hash}\n"
                        f"  保存されたハッシュ: {self._stored_hash}"
                    )
                else:
                    print("\n設定ファイルのハッシュ値は一致しています。整合性は確認されました。")
        else:
            # Case: Config file did NOT exist initially (it was just created).
            # We should save the current hash as the new stored hash.
            print(f"設定ファイル {self.file_path} が新規作成されました。現在のハッシュを保存します。")
            try:
                 self._hash_manager.save_hash(self._current_file_hash, self.hash_file_path)
                 # After saving, the stored_hash should now be the current hash.
                 # For consistency and potential future use, we can re-load it,
                 # but it's not strictly necessary for the initial creation case's logic flow.
                 # Let's update the attribute directly as we know what we just saved.
                 self._stored_hash = self._current_file_hash
                 print(f"現在のハッシュ {self._current_file_hash} を {self.hash_file_path} に保存しました。")

            except IOError as e:
                 # If saving the hash file fails during initial creation, this is a critical error.
                 # The config file was created, but its integrity cannot be tracked.
                 # It might be better to clean up the config file and raise an error.
                 print(f"致命的エラー: 新規設定ファイル '{self.file_path}' のハッシュ値の保存に失敗しました: {e}")
                 # Attempt to clean up the newly created config file as it's now untracked.
                 if os.path.exists(self.file_path):
                      try:
                           os.remove(self.file_path)
                           print(f"エラー発生のため、新規作成された設定ファイル '{self.file_path}' を削除しました。")
                      except Exception as cleanup_e:
                           print(f"エラー発生後のクリーンアップに失敗しました: {cleanup_e}")
                 raise IOError(f"致命的エラー: 新規設定ファイル '{self.file_path}' のハッシュ値の保存に失敗しました: {e}")


    # Step 9: Provide access to configuration data
    def get_setting(self, key, default=None):
        """
        設定データからキーに対応する値を取得する。

        Args:
            key (str): 取得したい設定値のキー。
            default (any, optional): キーが存在しない場合に返すデフォルト値。Defaults to None.

        Returns:
            any: 設定値、またはキーが存在しない場合はデフォルト値。
        """
        return self.data.get(key, default)

    # Step 10: Methods to get hash values
    def get_current_file_hash(self):
        """
        ConfigManagerが読み込み時に計算した、現在の設定ファイルのハッシュ値を取得する。

        Returns:
            str or None: 現在のファイルハッシュ値、またはエラー等で計算できていない場合はNone。
        """
        return self._current_file_hash

    def get_stored_hash(self):
        """
        対応するハッシュファイルから読み込まれたハッシュ値を取得する。

        Returns:
            str or None: ハッシュファイルから読み込まれたハッシュ値、またはファイルが存在しない場合はNone。
        """
        return self._stored_hash

    # Step 11: __str__ method
    def __str__(self):
        """
        設定データのJSON文字列表現を返す。

        Returns:
            str: 設定データの整形されたJSON文字列。
        """
        return json.dumps(self.data, indent=4)

class HashManager:
    """
    ハッシュ値の計算と、ハッシュ値をファイルに保存/読み込みするクラス。
    """
    def calculate_hash(self, content):
        """
        文字列またはバイト列のSHA256ハッシュ値を計算する。

        Args:
            content (str or bytes): ハッシュ値を計算する対象のデータ。

        Returns:
            str: 計算されたSHA256ハッシュ値の16進数表現。

        Raises:
            TypeError: contentが文字列またはバイト列でない場合。
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        elif not isinstance(content, bytes):
            raise TypeError("Content must be a string or bytes.")
        return hashlib.sha256(content).hexdigest()

    def save_hash(self, hash_value, hash_file_path):
        """
        ハッシュ値を指定されたファイルに保存する。

        Args:
            hash_value (str): 保存するハッシュ値。
            hash_file_path (str): ハッシュファイルを保存するパス。

        Raises:
            IOError: ファイルの保存中にエラーが発生した場合。
        """
        try:
            with open(hash_file_path, "w") as f:
                f.write(hash_value)
            print(f"ハッシュ値を {hash_file_path} に保存しました。")
        except IOError as e:
            print(f"エラー: ハッシュファイル '{hash_file_path}' の保存中にエラーが発生しました: {e}")
            raise # エラーを再度発生させる

    def load_hash(self, hash_file_path):
        """
        指定されたハッシュファイルからハッシュ値を読み込む。
        ファイルが存在しない場合はNoneを返す。

        Args:
            hash_file_path (str): ハッシュファイルを読み込むパス。

        Returns:
            str or None: 読み込まれたハッシュ値、またはファイルが存在しない場合はNone。

        Raises:
            IOError: ファイルの読み込み中にエラーが発生した場合。
        """
        if not os.path.exists(hash_file_path):
            print(f"ハッシュファイル {hash_file_path} が見つかりませんでした。")
            return None

        try:
            with open(hash_file_path, "r") as f:
                stored_hash = f.read().strip()
            print(f"ハッシュ値を {hash_file_path} から読み込みました。")
            return stored_hash
        except IOError as e:
            print(f"エラー: ハッシュファイル '{hash_file_path}' の読み込み中にエラーが発生しました: {e}")
            raise # エラーを再度発生させる

class ConfigLoader:
    """
    設定ファイルを読み込み、JSONとしてパースするクラス。
    ファイルが存在しない場合は新規作成機能も提供する。
    """
    def __init__(self, file_path):
        """
        ConfigLoaderの新しいインスタンスを初期化する。

        Args:
            file_path (str): 設定ファイルのパス。
        """
        self.file_path = file_path
        self.data = {}

    def _load_config(self):
        """
        設定ファイルを読み込み、JSONとしてパースする内部メソッド。
        ファイルが存在しない場合や読み込み/パースに失敗した場合は例外を発生させる。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"エラー: 設定ファイル '{self.file_path}' が見つかりません。")

        try:
            with open(self.file_path, "r") as f:
                file_content = f.read()
                self.data = json.loads(file_content)
            print(f"設定ファイルを {self.file_path} から読み込みました。")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"エラー: 設定ファイル '{self.file_path}' のJSON形式が不正です。", self.file_path, 0)
        except IOError as e:
            raise IOError(f"エラー: 設定ファイル '{self.file_path}' の読み込み中にエラーが発生しました: {e}")


    def _create_default_config(self):
        """
        デフォルト設定データを使用して新しい設定ファイルを作成する内部メソッド。
        ファイル作成に失敗した場合は例外を発生させる。
        """
        default_data = {
            "data1-str": "Default Value",
            "data2-float": 1.0,
            "data3-int": 100,
            "data4-bool": False
        }
        file_content = json.dumps(default_data, indent=4)
        try:
            with open(self.file_path, "w") as f:
                f.write(file_content)
            self.data = default_data
            print(f"デフォルト設定を {self.file_path} に書き込みました。")
        except IOError as e:
            raise IOError(f"エラー: デフォルト設定ファイル '{self.file_path}' の作成に失敗しました: {e}")


    def load_or_create(self):
        """
        ファイルが存在すれば読み込み、存在しなければ新規作成する。
        """
        if os.path.exists(self.file_path):
            print(f"{self.file_path} が存在します。読み込みを試みます。")
            self._load_config()
        else:
            print(f"{self.file_path} が存在しません。新規作成します。")
            self._create_default_config()

    def get_data(self):
        """
        読み込まれた、または作成された設定データ（辞書形式）を取得する。

        Returns:
            dict: 設定データ。
        """
        return self.data

