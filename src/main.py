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

import argparse
import torch
import json
import os
import shutil

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def load_presets(file_path="presets.json", sample_path="presets.json.sample"):
    """
    設定ファイルを読み込む。
    実体がない場合はサンプルからコピーを試みる。
    """
    # 1. presets.json がなくて sample がある場合はコピーしてあげる（親切設計）
    if not os.path.exists(file_path) and os.path.exists(sample_path):
        print(f"[*] {file_path} not found. Creating from {sample_path}...")
        shutil.copy(sample_path, file_path)

    # 2. それでもファイルがない場合は空の辞書を返す
    if not os.path.exists(file_path):
        print("[!] Warning: No preset file found. Using internal defaults.")
        return {}

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Error: {file_path} is not a valid JSON. Ignoring presets.")
            return {}

    # エイリアス展開
    presets = {k: v for k, v in data.items() if k != "aliases"}
    aliases = data.get("aliases", {})
    for alias, target in aliases.items():
        if target in presets:
            presets[alias] = presets[target]
            
    return presets

def parse_args():
    # プリセットデータの読み込み
    PRESETS = load_presets("presets.json")

    parser = argparse.ArgumentParser(description="Multi-Agent RL Parser with Presets")

    # --- 1. プリセット選択引数 ---
    # choicesに辞書のキーを渡すことで、存在しないプリセット指定をエラーにできる
    parser.add_argument('-P', '--preset', choices=PRESETS.keys(), help="Use a predefined preset from JSON")

    # --- 既存の引数 ---
    parser.add_argument('-g','--grid_size', default=4, type=int)
    parser.add_argument('-A','--agents_number', default=2, type=int)
    parser.add_argument('-G','--goals_number', default=2, type=int)
    parser.add_argument('-l','--learning_mode', choices=['Q', 'DQN'], default='DQN')
    parser.add_argument('--optimizer', choices=['Adam', 'RMSProp'], default='Adam')
    parser.add_argument('--mask', choices=[0, 1], default=0, type=int)
    parser.add_argument('--neighbor_distance', default=256, type=int)# 大きい値にしておくことで全観測になる。
    parser.add_argument('--reward_mode', choices=[0, 1, 2, 3], default=2, type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('-e','--episode_number', default=1200, type=int)
    parser.add_argument('-T','--max_timestep', default=150, type=int)
    parser.add_argument('--epsilon_decay', default=0.50, type=float)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--save_agent_states', choices=[0, 1], default=1, type=int)
    parser.add_argument('--window_width', default=500, type=int)
    parser.add_argument('--window_height', default=500, type=int)
    parser.add_argument('--render_mode', choices=[0, 1], default=0, type=int)
    parser.add_argument('--pause_duration', default=0.1, type=float)
    parser.add_argument('--target_update_frequency', default=500, type=int)
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--beta', default=0.4, type=float)
    parser.add_argument('--beta_anneal_steps', default=20000, type=int)
    parser.add_argument('--use_per', choices=[0, 1], default=0, type=int)

    # 3. 1回目のパース：プリセットの有無だけを確認
    # args, unknown = parser.parse_known_args() ではなく、
    # プリセット引数だけを抽出してチェックする
    temp_args, _ = parser.parse_known_args()
    
    if temp_args.preset:
        target = PRESETS[temp_args.preset]
        if isinstance(target, str):
            target = PRESETS[target]
        
        # プリセットをデフォルト値として設定
        parser.set_defaults(**target)
    
    # 4. 最終的なパース
    # ここで sys.argv (または引数なし) を使うことで、
    # 「プリセットで上書きされたデフォルト値」を「個別のコマンド引数」がさらに上書きできる
    args = parser.parse_args() 
    
    return args

if __name__ == '__main__':
    config = parse_args()
    print(f"Loaded config: {config}")

    # auto選択時のデバイス決定ロジックを追加
    if config.device == 'auto':
        if torch.cuda.is_available():
            config.device = 'cuda'
        elif torch.backends.mps.is_available():
            config.device = 'mps'
        else:
            config.device = 'cpu'
        print(f"自動選択されたデバイス: {GREEN}{config.device}{RESET}\n")

    def q_learning():
        from Q_learn.MultiAgent_Q import MultiAgent_Q
        from Q_learn.Agent_Q import Agent
        agents:list[Agent] = [Agent(config,id) for id in range(config.agents_number)]
        simulation = MultiAgent_Q(config,agents)
        
        simulation.train(config.episode_number)

        simulation.result_save()

        simulation.render_anime(config.episode_number)
        #print("reward:",r, "done:",done)
        #print("GET: trajectry")
        #for tr in traj: print(tr)
    
    def q_play():
        from Q_learn.MultiAgent_Q import MultiAgent_Q
        from Q_learn.Agent_Q import Agent
        agents:list[Agent] = [Agent(config,id) for id in range(config.agents_number)]
        simulation = MultiAgent_Q(config,agents)

        #simulation.render_anime(config.episode_number)
        # simulation.log_disp()
            

    def dqn_process():
        from DQN.MultiAgent_DQN import MARLTrainer
        from DQN.Agent_DQN import Agent
        agents:list = [Agent(id, config,config.use_per) for id in range(config.agents_number)]

        simulation = MARLTrainer(config,agents)

        simulation.train()

        simulation.result_save()

        simulation.save_model_weights()
        simulation.load_model_weights()

        simulation.simulate_agent_behavior(max_simulation_timestep=150)
    
    def dimensions_estimater(grid_size:int, agent_number:int)->int:
        res = 1
        for i in range(agent_number):
            res *= (grid_size * grid_size - i)

        return res

    if config.learning_mode == "Q":
        if dimensions_estimater(config.grid_size, config.agents_number)>1e6: 
            raise ValueError(f"警告:推定空間サイズ({dimensions_estimater(config.grid_size, config.agents_number)})が大きすぎます")
        
        if config.episode_number == -1:
            q_play()
        else:
            q_learning()

    elif config.learning_mode == "DQN":
        dqn_process()
    else:
        print("未実装\n")