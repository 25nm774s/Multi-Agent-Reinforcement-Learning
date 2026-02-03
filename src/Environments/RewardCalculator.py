from itertools import product
from typing import Dict

class RewardCalculator:
    def __init__(self, grid_size, reward_mode, agent_ids, agents_number, goals_number) -> None:
        self.grid_size = grid_size
        self.reward_mode =reward_mode
        self._agent_ids = agent_ids
        self.agents_number = agents_number
        self.goals_number = goals_number

        self.prev_distances: Dict[str, float] = {aid: float('inf') for aid in self._agent_ids}

    def _calculate_total_distance_to_goals(self, agent_positions, goal_positions) -> Dict[str, float]:
        """
        エージェントとゴールの最適なペアリングを行い、
        エージェントIDをキーとしたマンハッタン距離の辞書を返します。
        """
        # 1. 計算量のチェック
        if self.agents_number * self.goals_number > 400:
            raise Exception(f"[警告] 計算量のオーバーヘッドが懸念されます。ペア数: {self.agents_number * self.goals_number}")

        # 初期値として全エージェントを inf で埋めた辞書を作成
        distances = {aid: float('inf') for aid in self._agent_ids}

        if not agent_positions or not goal_positions:
            return distances

        # 2. 全ペアの距離を計算（IDを保持したままリスト化）
        # (dist, aid, gid) のタプルのリストを作る
        all_pairs = []
        for (aid, a_pos), (gid, g_pos) in product(agent_positions.items(), goal_positions.items()):
            dist = abs(a_pos[0] - g_pos[0]) + abs(a_pos[1] - g_pos[1])
            all_pairs.append((dist, aid, gid))

        # 3. 距離が近い順にソート
        all_pairs.sort(key=lambda x: x[0])

        # 4. 近いペアから順に確定
        used_agents = set()
        used_goals = set()
        limit = min(len(agent_positions), len(goal_positions))

        for dist, aid, gid in all_pairs:
            if aid not in used_agents and gid not in used_goals:
                distances[aid] = float(dist)
                used_agents.add(aid)
                used_goals.add(gid)

                if len(used_agents) == limit:
                    break

        return distances    
    
    def _calculate_reward(self, done, agent_pos, goal_pos) -> Dict[str,float]:
        """
        現在の状態と報酬モードに基づいて報酬を計算します。
        """
        reward_dict: Dict[str, float] = {}
        
        # 共通設定
        max_penalty_dist = float(self.grid_size * 2) # inf時のクリッピング用

        if self.reward_mode == 0:
            # モード 0: 完了条件を満たしていれば +10、それ以外 0
            # reward_dict = 10.0 if done else 0.0
            for aid in self._agent_ids:
                reward_dict[aid] = 10.0 if done else 0.0

        elif self.reward_mode == 1:
            # モード 2: 完了条件を満たしていれば +10、それ以外 -0.02
            for aid in self._agent_ids:
                reward_dict[aid] = 10.0 if done else -0.02

        elif self.reward_mode == 2:
            distances_dict = self._calculate_total_distance_to_goals(agent_pos, goal_pos)
            for aid in self._agent_ids:
                dist = distances_dict.get(aid, float('inf'))
                reward_dict[aid] = -min(dist, max_penalty_dist)

        elif self.reward_mode == 3:
            # --- モード 3: Potential-based Reward Shaping ---
            current_distances = self._calculate_total_distance_to_goals(agent_pos, goal_pos)
            
            for aid in self._agent_ids:
                # 1. 距離の取得とクリッピング (inf対策)
                prev_d = min(self.prev_distances.get(aid, max_penalty_dist), max_penalty_dist)
                curr_d = min(current_distances.get(aid, max_penalty_dist), max_penalty_dist)
                
                # 2. ポテンシャル報酬: (前回の距離 - 今回の距離) 
                # 近づけばプラス、遠ざかればマイナス
                shaping_reward = float(prev_d - curr_d) * 0.05
                
                # 3. ステップペナルティ (早く全員揃うことを促す)
                step_penalty = -0.01
                
                # 4. 完了ボーナス (全員同時にゴールにいる場合)
                completion_bonus = 1.0 if done else 0.0
                
                reward_dict[aid] = shaping_reward + step_penalty + completion_bonus
            
            # 5. prev_distances を更新 (次のステップ用)
            self.prev_distances = current_distances
        elif self.reward_mode == 4:
            # --- モード 3: Potential-based Reward Shaping ---
            current_distances = self._calculate_total_distance_to_goals(agent_pos, goal_pos)
            
            for aid in self._agent_ids:
                # 1. 距離の取得とクリッピング (inf対策)
                prev_d = min(self.prev_distances.get(aid, max_penalty_dist), max_penalty_dist)
                curr_d = min(current_distances.get(aid, max_penalty_dist), max_penalty_dist)
                
                # 2. ポテンシャル報酬: (前回の距離 - 今回の距離) 
                # 近づけばプラス、遠ざかればマイナス
                shaping_reward = float(prev_d - curr_d)
                
                # 3. ステップペナルティ (早く全員揃うことを促す)
                step_penalty = -0.1
                
                # 4. 完了ボーナス (全員同時にゴールにいる場合)
                
                reward_dict[aid] = shaping_reward + step_penalty
            
            # 5. prev_distances を更新 (次のステップ用)
            self.prev_distances = current_distances

        else:
            print(f"Warning: 未知の報酬モード: {self.reward_mode}。報酬は 0 です。")
            reward_dict = {}
            for aid in self._agent_ids:
                reward_dict[aid] = 0.0

        return reward_dict
