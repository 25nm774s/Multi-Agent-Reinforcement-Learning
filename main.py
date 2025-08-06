"""
このファイルを実行して学習する．
python main.py --grid_size 10 のようにして各種設定を変更可能.
主要なハイパーパラメータは parse_args() で管理.

ゴール位置は初期に一度だけランダム生成し，エージェント位置のみエピソード毎に再生成するように修正済み。

--- リファクタリングに関する議論のまとめ (2024/05/20) ---
現在の Main クラスの処理を、よりオブジェクト指向的に構造化する方向性を議論。
主要な提案は以下の通り：
1.  マルチエージェントシステムの中核ロジックを MultiAgent クラスにカプセル化。
2.  MultiAgent クラス内に train() と evaluate() メソッドを分離。
3.  さらに進んで、学習アルゴリズム (Q, DQNなど) を AgentBase を継承したサブクラスとして実装し、MultiAgent がこれらを管理する（ポリモーフィズム活用）。

メリット：責務明確化、コード整理、再利用性/拡張性向上、テスト容易性、状態管理改善。
考慮点：学習/評価の処理分離、状態管理、引数渡し、モデル保存/読み込み、ログ/可視化、学習モードの扱い、共通インターフェース設計（サブクラス化の場合）。
目的：コードの保守性・拡張性を高め、将来的な機能追加（例: 新しいアルゴリズム）を容易にする。
---------------------------------------------------------

TODO: 学習モードごとの処理分岐(if self.learning_mode == ...)がMainクラスのrunメソッド内に混在しており、
      保守性・拡張性の観点から、学習モードごとに別のクラス(例: Q_Main, DQN_Main)に分割する設計を検討する。
      全体を制御するクラスで、どの学習モードのクラスを使うかを選択する形にする。

"""

# --- プログレスバー（進捗表示）に関する注意点と解決策 ---
# Colab環境ではリアルタイムに表示されるプログレスバー（例: '■'）が、
# ローカルPython環境で実行すると、処理が完了した後に一気に表示されてしまう場合があります。
# これは、Pythonの標準出力がパフォーマンス向上のために「バッファリング」されるためです。

# この問題を解決し、ローカル環境でもリアルタイムにプログレスバーを表示するための方法は以下の通りです。

# 1. print()関数の 'flush=True' 引数を使用する (最もシンプル)
#    - print()関数に 'flush=True' を追加すると、出力が即座に画面に書き出されます。
#    - 例: print('■', end='', flush=True)

# 2. sys.stdout.flush() を使用する (より柔軟な制御が必要な場合)
#    - print()以外の方法で出力している場合や、特定のタイミングでまとめてフラッシュしたい場合に有効です。
#    - import sys をファイルの先頭に追加し、出力後に sys.stdout.flush() を呼び出します。
#    - 例:
#      import sys
#      sys.stdout.write('■')
#      sys.stdout.flush()

# 3. tqdm ライブラリを使用する (推奨: より高機能で美しいプログレスバー)
#    - プログレスバーの表示に特化した外部ライブラリです。
#    - 内部で適切なフラッシュ処理が行われるため、Colabでもローカルでも期待通りに動作します。
#    - 残り時間推定などの追加機能も提供されます。
#    - インストール: pip install tqdm
#    - 使用例:
#      from tqdm import tqdm
#      for item in tqdm(iterable_object):
#          # 処理内容
#          pass
# --------------------------------------------------------

import argparse
import torch

from agent import Agent

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir_path', default='./')
        parser.add_argument('--grid_size', default=4, type=int)
        parser.add_argument('--agents_number', default=2, type=int)
        parser.add_argument('--goals_number', default=2, type=int)
        parser.add_argument('--learning_mode', choices=['Q', 'DQN'], default='DQN')
        parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
        parser.add_argument('--mask', choices=[0, 1], default=0, type=int)
        parser.add_argument('--load_model', choices=[0, 1], default=0, type=int)
        parser.add_argument('--reward_mode', choices=[0, 1, 2], default=0, type=int)
        parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
        parser.add_argument('--episode_number', default=5000, type=int)
        parser.add_argument('--max_timestep', default=25, type=int)
        parser.add_argument('--decay_epsilon', default=500000, type=int)
        parser.add_argument('--learning_rate', default=0.000005, type=float)
        parser.add_argument('--gamma', default=0.99, type=float)
        parser.add_argument('--buffer_size', default=10000, type=int)
        parser.add_argument('--batch_size', default=2, type=int)
        parser.add_argument('--save_agent_states', choices=[0, 1], default=1, type=int)
        parser.add_argument('--pause_duration', default=0.1, type=float)
        parser.add_argument('--render_mode', choices=[0, 1], default=0, type=int)
        return parser.parse_args()

    args = parse_args()

    # auto選択時のデバイス決定ロジックを追加
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
        print(f"自動選択されたデバイス: {GREEN}{args.device}{RESET}\n")

    def q_learning():
        from Q_learn.MultiAgent_Q import MultiAgent_Q
        config = parse_args()
        agents:list = [Agent(config,id) for id in range(config.agents_number)]
        simulation = MultiAgent_Q(config,agents)

        simulation.run()

        simulation.result_save()

    q_learning()