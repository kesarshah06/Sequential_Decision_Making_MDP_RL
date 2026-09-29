import numpy as np
import time
from env import HighwayEnv

# Action constants
ACTION_INCREASE_SPEED = 0
ACTION_DECREASE_SPEED = 1
ACTION_INCREASE_LANE = 2
ACTION_DECREASE_LANE = 3
ACTION_NO_OP = 4

# ........................ Q-Learning Highway Agent ........................
class Agent:

    # .................... Initialize Agent and Q-Table ....................
    def __init__(self, env: HighwayEnv, discount_factor = 0.99):
        self.env = env
        self.discount_factor = discount_factor

        self.num_speeds = 4
        self.num_lanes = 4
        self.dist_states = 5
        self.num_states = self.num_speeds * self.num_lanes * (self.dist_states ** 4)    # Total 10000 states
        self.num_actions = 5

        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        self.alpha_initial = 0.25
        self.alpha_min = 0.05
        self.epsilon_min = 0.01

        self.pirate_delta_metric = 0.0
        self.treasure_highway_matrix = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

    # .................... Encode State Tuple to Index .....................
    def encode_state(self, speed, lane, min_dist) -> int:
        speed_idx = int(speed)
        lane_idx = int(lane)
        d0 = int(min_dist[0])
        d1 = int(min_dist[1])
        d2 = int(min_dist[2])
        d3 = int(min_dist[3])

        return speed_idx * 2500 + lane_idx * 625 + d0 * 125 + d1 * 25 + d2 * 5 + d3    # Compute 1D state index

    # .................. Epsilon-Greedy Action Selection ...................
    def select_action(self, state_idx: int, speed: int, lane: int, min_dist: list, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            dist_ahead = min_dist[lane]
            action = np.random.randint(self.num_actions)
            if dist_ahead == 1 and action in (ACTION_INCREASE_SPEED, ACTION_NO_OP):    # Avoid suicide action when blocked
                action = np.random.choice([ACTION_DECREASE_SPEED, ACTION_INCREASE_LANE, ACTION_DECREASE_LANE])
            return action
        else:
            return self._greedy_action(state_idx, speed, lane, min_dist)

    # ...................... Greedy Action Selection .......................
    def _greedy_action(self, state_idx: int, speed: int, lane: int, min_dist: list) -> int:
        q_vals = self.q_table[state_idx].copy()
        dist_ahead = min_dist[lane]

        if dist_ahead == 1:                                                 # Apply safety heuristic for obstacle ahead
            q_vals[ACTION_INCREASE_SPEED] -= 5.0
            q_vals[ACTION_NO_OP] -= 3.0

            if lane < 3 and min_dist[lane + 1] in (0, 4, 3):
                q_vals[ACTION_INCREASE_LANE] += 2.0
            if lane > 0 and min_dist[lane - 1] in (0, 4, 3):
                q_vals[ACTION_DECREASE_LANE] += 2.0
            q_vals[ACTION_DECREASE_SPEED] += 1.0
        elif dist_ahead in (0, 4, 3):                                       # Clear lane ahead: prefer accelerating
            if speed < 3:
                q_vals[ACTION_INCREASE_SPEED] += 1.5
            else:
                q_vals[ACTION_NO_OP] += 1.0

        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    # ....................... Update Q-Value TD Step .......................
    def update_q_value(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool, alpha: float, speed: int, lane: int, min_dist: list) -> float:
        shaped_reward = reward
        dist_ahead = min_dist[lane]

        if done and reward < 0:
            shaped_reward = -10.0                                           # Penalty for collision
        elif dist_ahead == 1:                                               # Reward shaping for imminent danger
            if action in (ACTION_INCREASE_SPEED, ACTION_NO_OP):
                shaped_reward -= 2.0
            elif action == ACTION_DECREASE_SPEED:
                shaped_reward += 0.5
            elif action == ACTION_INCREASE_LANE and lane < 3 and min_dist[lane + 1] != 1:
                shaped_reward += 1.0
            elif action == ACTION_DECREASE_LANE and lane > 0 and min_dist[lane - 1] != 1:
                shaped_reward += 1.0

        if done:
            target = shaped_reward
        else:
            target = shaped_reward + self.discount_factor * np.max(self.q_table[next_state_idx])    # Bellman target calculation

        delta = target - self.q_table[state_idx, action]
        self.q_table[state_idx, action] += alpha * delta                    # Q-learning TD update

        self.pirate_delta_metric = max(self.pirate_delta_metric, abs(delta))
        self.treasure_highway_matrix[state_idx, action] += 1.0

        return delta

    # .................. Check Treasure Threshold Metric ...................
    def check_treasure_threshold(self, delta: float, limit: float) -> bool:
        return delta <= limit

    # ................... Calculate Lane Entropy Metric ....................
    def calculate_lane_entropy(self, ship_location, pirate_locations) -> float:
        if not pirate_locations:
            return 0.0
        dists = [abs(ship_location[0] - p[0]) + abs(ship_location[1] - p[1]) for p in pirate_locations]
        probs = np.array(dists, dtype=np.float64) / (sum(dists) + 1e-8)
        return float(-np.sum(probs * np.log(probs + 1e-8)))

    # ...................... Train Q-Learning Policy .......................
    def learn_policy(self, time_limit: float):
        start_time = time.time()
        max_duration = max(0.1, time_limit - 0.4)

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break

            progress = min(1.0, elapsed / max_duration)
            epsilon = max(self.epsilon_min, (1.0 - progress) ** 1.5)        # Time-proportional exploration decay
            alpha = max(self.alpha_min, self.alpha_initial * (1.0 - 0.7 * progress))    # Learning rate decay

            state = self.env.reset()
            speed, lane, min_dist = state
            state_idx = self.encode_state(speed, lane, min_dist)
            done = False

            while not done:
                if time.time() - start_time >= max_duration:
                    break

                action = self.select_action(state_idx, speed, lane, min_dist, epsilon)
                next_state, reward, done = self.env.step(action)
                next_speed, next_lane, next_min_dist = next_state
                next_state_idx = self.encode_state(next_speed, next_lane, next_min_dist)

                self.update_q_value(state_idx, action, reward, next_state_idx, done, alpha, speed, lane, min_dist)
                
                state = next_state
                speed, lane, min_dist = next_speed, next_lane, next_min_dist
                state_idx = next_state_idx

    # ......................... Get Greedy Action ..........................
    def get_action(self, speed, lane, min_dist) -> int:
        state_idx = self.encode_state(speed, lane, min_dist)
        return self._greedy_action(state_idx, speed, lane, min_dist)
