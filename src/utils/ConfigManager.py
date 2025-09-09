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
    def __init__(self, initial_data:dict, config_dir="folder"):
        """
        ConfigManagerの新しいインスタンスを初期化する。
        指定されたファイルパスから設定とハッシュを読み込み、整合性をチェックする。
        設定ファイルが存在しない場合は新規作成し、そのハッシュを保存する。

        Args:
            config_dir (str): 設定ファイルを置くフォルダ

        Raises:
            FileNotFoundError: 設定ファイルが存在しない場合 (ConfigLoaderが処理)。
            json.JSONDecodeError: 設定ファイルのJSON形式が不正な場合 (ConfigLoaderが処理)。
            IOError: ファイルの読み書き中にエラーが発生した場合、または設定ファイルは存在するがハッシュファイルが見つからない場合。
            ValueError: 設定ファイルの内容と保存されたハッシュ値が一致しない場合。
            Exception: 上記以外の予期しないエラーが発生した場合。
        """
        #config_dir = ("folder")
        self.json_file_path = os.path.join(config_dir, "config_j.json")
        self.hash_file_path = os.path.join(config_dir, "config_h.hash")
        self.data = {}
        self._current_file_hash = None
        self._stored_hash = None

        # ヘルパークラスをインスタンス化
        self._config_loader = ConfigLoader(self.json_file_path)
        self._hash_manager = HashManager()

        # チェック1: フォルダが存在...config(json/hash)が存在
        # チェック1: フォルダがない...新規として扱い、configを作成(save)。
        if not os.path.exists(config_dir):
            print(f"フォルダ{config_dir}がないため、新規として扱い、configを発行。")
            self._create_process(config_dir, initial_data)
        #else:
        #    print(f"フォルダ{config_dir}が存在するため、ロードを試みます。")
        #    self._load_process()

        # 新規が作られたタイミングでロード
        self._load_process()

        # チェック2.1: jsonファイルのハッシュを計算
        try:    #  1. ファイルの内容を取得
            # ファイルを再度読み込み、生のファイルコンテンツを取得
            with open(self.json_file_path, "r") as f:
                file_content_str = f.read()

            # 2. ハッシュを計算（生のファイルコンテンツから）
            self._current_file_hash = self._hash_manager.calculate_hash(file_content_str)
            #print(f"Calculated hash from file content: {self._current_file_hash}") # デバッグ用


        except IOError as e:
            # _config_loader.load_or_create() が成功していれば、これは発生しないはず
            # 設定ファイルが最初に存在していた場合、ConfigLoader.load_or_create は読み取りエラーを処理しているはず
            # 設定ファイルが新しく作成された場合、ConfigLoader.load_or_create は書き込みエラーを処理しているはず
            # これは、読み込み/作成後、ハッシュのために再度読み取るまでのまれなタイミングの問題や権限の変更をキャッチする可能性がある
            raise IOError(f"エラー: 読み込み/作成後のファイル内容の読み取り中にエラーが発生しました: {e}")

        except TypeError as e:
            raise TypeError(f"エラー: ハッシュ値の計算中に型エラーが発生しました: {e}")


        # チェック2.2: ハッシュを記録したファイルを参照
        try:
            self._stored_hash = self._hash_manager.load_hash(self.hash_file_path)
        except FileNotFoundError as e:
            # HashManager.load_hash が FileNotFoundError を発生させた (例: ハッシュファイルがない)
            raise #FileNotFoundError(f"エラー: ハッシュファイル '{self.hash_file_path}' が見つかりません: {e}")
        except IOError as e:
            # HashManager.load_hash が IOError を発生させた (例: ハッシュファイルの読み取り権限エラー)
            raise #IOError(f"エラー: ハッシュファイル '{self.hash_file_path}' の読み込み中にエラーが発生しました: {e}")


        # 設定ファイルが存在し、ハッシュファイルも存在する場合、ハッシュ値を比較する
        if self._current_file_hash == self._stored_hash:
            print("\n設定ファイルのハッシュ値は一致しています。整合性は確認されました。")
            # ハッシュが一致した場合のみデータをロード
            self.data = self._config_loader.get_data()
            print(f"設定データ: \n{self.data}")
        else:
            # Step 7: 不一致の場合は ValueError を発生させる
            raise ValueError(
                f"エラー: 設定ファイル '{self.json_file_path}' の内容が変更された可能性があります！\n"
                f"  現在のハッシュ    : {self._current_file_hash}\n"
                f"  保存されたハッシュ: {self._stored_hash}"
            )

    def _create_process(self, config_dir, initial_data:dict):
        """
        新規として扱う場合呼び出され、configを発行する。
        """
        os.makedirs(config_dir)
        
        print(f"initial_data.type: {type(initial_data)}")

        # ConfigLoader を使用してファイルを作成
        try:
            self._config_loader.create_config(initial_data)
        except Exception as e:
            print(f"エラー: 設定ファイル '{self.json_file_path}' の作成中にエラーが発生しました: {e}")
            raise # 元の例外を再度発生させる

        # 作成されたファイル内容を読み込んでハッシュを計算
        try:
            with open(self.json_file_path, "r") as f:
                file_content_str = f.read()
            self._current_file_hash = self._hash_manager.calculate_hash(file_content_str)
            #print(f"Calculated hash after creation: {self._current_file_hash}") # デバッグ用
        except IOError as e:
            raise IOError(f"エラー: 作成されたファイル内容の読み取り中にエラーが発生しました: {e}")
        except TypeError as e:
            raise TypeError(f"エラー: 作成後のハッシュ値計算中に型エラーが発生しました: {e}")


        # 計算したハッシュ値をハッシュファイルに保存
        try:
            self._hash_manager.save_hash(self._current_file_hash, self.hash_file_path)
        except IOError as e:
            raise IOError(f"エラー: ハッシュファイル '{self.hash_file_path}' の保存中にエラーが発生しました: {e}")


    def _load_process(self):
        try:
            self._config_loader.load_config()
        except Exception as e:
            print(f"エラー: 設定ファイル '{self.json_file_path}' の読み込み中にエラーが発生しました: {e}")
            raise # 元の例外を再度発生させる

    # Step 9: 設定データへのアクセスを提供する
    def get_setting(self, key, default=None):
        """
        設定データからキーに対応する値を取得する。

        Args:
            key (str): 取得したい設定値のキー。
            default (any, optional): キーが存在しない場合に返すデフォルト値。Defaults to None.

        Returns:
            any: 設定値、またはキーが存在しない場合はデフォルト値。
        """
        print("")
        return self.data.get(key, default)

    # Step 10: ハッシュ値を取得するメソッド
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

    # Step 11: __str__ メソッド
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

    def load_hash(self, hash_file_path) -> str:
        """
        指定されたハッシュファイルからハッシュ値を読み込む。。

        Args:
            hash_file_path (str): ハッシュファイルを読み込むパス。

        Returns:
            str: 読み込まれたハッシュ値

        Raises:
            IOError: ファイルの読み込み中にエラーが発生した場合。
            FileNotFoundError ファイルが存在しない場合。
        """
        if not os.path.exists(hash_file_path):
            #print(f"ハッシュファイル {hash_file_path} が見つかりませんでした。")
            raise FileNotFoundError(f"エラー: ハッシュファイル '{hash_file_path}' が見つかりません。")

        try:
            with open(hash_file_path, "r") as f:
                stored_hash = f.read().strip()
            print(f"ハッシュ値を {hash_file_path} から読み込みました。")
            return stored_hash
        except IOError as e:
            #print(f"エラー: ハッシュファイル '{hash_file_path}' の読み込み中にエラーが発生しました: {e}")
            raise OSError(f"エラー: ハッシュファイル '{hash_file_path}' の読み込み中にエラーが発生しました: {e}")

'''

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

    def load_config(self):
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


    def create_config(self, json_data:dict):
        """
        デフォルト設定データを使用して新しい設定ファイルを作成する内部メソッド。
        ファイル作成に失敗した場合は例外を発生させる。
        また、入力されたjson_dataが有効なJSON形式であるかチェックする。
        """
        default_data = {
            "data1-str": "Default Value",
            "data2-float": 1.0,
            "data3-int": 100,
            "data4-bool": False
        }

        input_data = default_data if json_data is None else json_data

        # JSON形式のチェック
        try:
            file_content = json.dumps(input_data, indent=4)
        except TypeError as e:
            raise TypeError(f"エラー: 提供されたデータは有効なJSON形式ではありません: {e}")

        try:
            with open(self.file_path, "w") as f:
                f.write(file_content)
            self.data = input_data # 作成されたデータを自身のデータとして保持
            print(f"設定を {self.file_path} に書き込みました。")
        except IOError as e:
            raise IOError(f"エラー: 設定ファイル '{self.file_path}' の作成に失敗しました: {e}")

    def get_data(self):
        """
        読み込まれた、または作成された設定データ（辞書形式）を取得する。

        Returns:
            dict: 設定データ。
        """
        return self.data

    def __str__(self) -> str:
        return json.dumps(self.data, indent=4)

'''


class ConfigLoader:
    """
    設定ファイルを読み込み、保存するためのクラス。
    JSON形式のファイルを扱い、キーを整数として管理します。
    """
    def __init__(self, file_path:str):
        """
        ConfigLoaderを初期化します。

        Args:
            file_path (str): 設定ファイルのパス。ファイルが存在しない場合は作成されます。
        """
        self.fpath  = file_path

        self._dir   = os.path.dirname(self.fpath)
        self._fname = os.path.basename(self.fpath)

        self._data = {}

        if os.path.exists(self.fpath):
            self._load()
        else:
            os.makedirs(self._dir, exist_ok=True)
            self._save(self.data)

    @property
    def data(self):
        """
        現在読み込まれている設定データを返します。

        Returns:
            dict: 設定データ。
        """
        return self._data

    def __len__(self):
        """
        設定データのキーの数を返します。

        Returns:
            int: キーの数。
        """
        return len(self._data)

    def __repr__(self):
        """
        設定データを整形されたJSON文字列として返します。

        Returns:
            str: 整形されたJSON文字列。
        """
        return json.dumps(self._data, indent=4)

    def __str__(self):
        """
        設定データを文字列として返します。

        Returns:
            str: 設定データの文字列表現。
        """
        return str(self._data)

    def __contains__(self, key):
        """
        指定されたキーが設定データに存在するかを確認します。

        Args:
            key: 確認するキー。

        Returns:
            bool: キーが存在する場合はTrue、それ以外はFalse。
        """
        return key in self._data

    def _save(self, json_data):
        """
        現在の設定データをファイルに保存します。

        Args:
            json_data (dict): 保存するデータ。
        """
        # Convert integer keys back to strings for saving
        string_keys_data = {str(k): v for k, v in json_data.items()}
        with open(self.fpath, "w") as file:
            json.dump(string_keys_data, file, indent=4)

    def _load(self):
        """
        ファイルから設定データを読み込みます。
        """
        if not os.path.exists(self.fpath):
            return

        with open(self.fpath, "r") as file:
            data = json.load(file)

        # Convert keys to integers after loading
        self._data = {str(k): v for k, v in data.items()}

    def update(self, key:str, value):
        """
        指定されたキーの値または新しいキーと値のペアを更新または追加します。

        Args:
            key (int): 更新または追加するキー（整数）。
            value (str): キーに対応する値（文字列）。

        Raises:
            ValueError: キーが整数でない、値が文字列でない、またはキーが負の値の場合。
        """
        #if not isinstance(key,int): raise ValueError('key type have to int')
        #if not isinstance(value,str): raise ValueError('value type have to str')

        #if key < 0: raise ValueError(f'Value Error: {key} not +0 -by update')

        # Check if the key exists using integer key
        if key in self.data:
            self._data[key] = value
        else:
            print(f"Info: key({key}) not exist")

            self._data[key] = value
            self._save(self.data)

    def add(self, key:str, value):
        """
        新しいキーと値のペアを追加します。キーが既に存在する場合はエラーとなります。

        Args:
            key (str): 追加するキー（変数名）。
            value (floar|int): キーに対応する値。

        Raises:
            ValueError: キーが整数でない、値が文字列でない、キーが負の値、またはキーが既に存在する場合。
        """
        #if not isinstance(key,int): raise ValueError('key type have to int')
        #if not isinstance(value,str): raise ValueError('value type have to str')

        #if key < 0: raise ValueError(f'Value Error: {key} not +0 -by add')

        if key in self.data:
            raise ValueError(f'key({key}) already exist')
        else:
            self._data[key] = value
            self._save(self.data)


if __name__=='__main__':
    import os

    check_suite = [0,0,0]
    dammy_path = os.path.join('./dammypath/data.json')

    cl = ConfigLoader(dammy_path)

    # 1回目...データは空
    if cl.data == {}:
        print(f"{cl.data} == (void)...期待通り")
        check_suite[0]=1

    # データを追加
    cl.add("data1", 7)
    cl.add("data2",42)

    del cl
    # 2回目...データに値が設定
    try:print(cl)#type:ignore
    except Exception: print("clインスタンスが存在しないことを確認.")
    cl = ConfigLoader(dammy_path)

    if cl.data == {"data1":7,"data2":42}:
        print(f"{cl.data} =='data1':7,'data2':42...期待通り")# データが表示されることを期待
        check_suite[1]=1

    # データを追加
    cl.update("data999", 999)

    del cl
    try:print(cl)#type:ignore
    except Exception: print("clインスタンスが存在しないことを確認.")

    cl = ConfigLoader(dammy_path)

    # 3回目...同様+追加データを期待
    if cl.data == {"data1":7,"data2":42,"data999":999}:
        print(f"{cl.data} =='data1':7,'data2':42,'data999',999...期待通り")
        check_suite[2]=1

    # サマリー表示
    if all(check_suite): print("全項目が期待通り終えました。")
    else: print(f"意図しない挙動あり。{check_suite}")

    try:
        os.remove(dammy_path)
        os.rmdir("./dammypath")
        print("ダミーパスを削除しました。")
    except Exception as e:
        raise e
