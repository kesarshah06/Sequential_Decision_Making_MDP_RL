import numpy as np
import time
from env import HighwayEnv

# Action constants
ACTION_INCREASE_SPEED = 0
ACTION_DECREASE_SPEED = 1
ACTION_INCREASE_LANE = 2
ACTION_DECREASE_LANE = 3
ACTION_NO_OP = 4

class Agent:
    """
    Model-Free Tabular Reinforcement Learning Agent for Highway Navigation.
    
    1. Temporal Difference (TD) Q-Learning Updates 
    2. Multi-Pass TD Trajectory Propagation for Data-Efficient Credit Assignment 
    3. Monte Carlo Discounted Return (G) Backpropagation for Terminal Crashes 
    4. Active RL with Epsilon-Greedy Strategy & Decaying Alpha/Epsilon 
    """

    def __init__(self, env: HighwayEnv, discount_factor=0.99):
        self.env = env
        self.discount_factor = discount_factor

        self.num_speeds = 4
        self.num_lanes = 4
        self.dist_states = 5
        # Total state space size: 4 speeds * 4 lanes * (5 distances ^ 4 lanes) = 10,000 states
        self.num_states = self.num_speeds * self.num_lanes * (self.dist_states ** 4)
        self.num_actions = 5

        # Q-Table Q(s, a) initialized to 0 (Slide 44)
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float64)

        # Learning Hyperparameters (Slides 45, 48)
        self.alpha_initial = 0.45
        self.alpha_min = 0.03
        self.epsilon_initial = 1.0
        self.epsilon_min = 0.01

    # .................... Encode State Tuple to Index .....................
    def encode_state(self, speed, lane, min_dist) -> int:
        return (int(speed) * 2500 + int(lane) * 625 +
                int(min_dist[0]) * 125 + int(min_dist[1]) * 25 +
                int(min_dist[2]) * 5 + int(min_dist[3]))

    # .................. Epsilon-Greedy Action Selection (Slide 48) ...................
    def select_action(self, state_idx: int, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.num_actions))
        return self._greedy_action(state_idx)

    # ...................... Greedy Action Selection (Slide 38) .......................
    def _greedy_action(self, state_idx: int) -> int:
        q_vals = self.q_table[state_idx]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        
        if len(best_actions) == 1:
            return int(best_actions[0])

        # Safety-first tie-breaking for unvisited / tied states:
        if max_val == 0.0:
            if ACTION_NO_OP in best_actions:
                return ACTION_NO_OP
            if ACTION_INCREASE_SPEED in best_actions:
                return ACTION_INCREASE_SPEED

        return int(np.random.choice(best_actions))

    # .................... Temporal Difference (TD) Step (Slides 28-31, 42-44) ....................
    def update_q_value(self, state_idx: int, action: int, reward: float, next_state_idx: int, done: bool, alpha: float) -> float:
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state_idx])

        # TD Error delta = target - Q(s, a) (Slide 31)
        delta = target - self.q_table[state_idx, action]
        # TD Q-value update (Slides 31, 44)
        self.q_table[state_idx, action] += alpha * delta
        return delta

    # ................. Train Agent using Lecture 09 Algorithms .................
    def learn_policy(self, time_limit: float):
        start_time = time.time()
        max_duration = max(0.1, time_limit - 0.35)
        df = self.discount_factor
        step_counter = 0

        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break

            progress = min(1.0, elapsed / max_duration)
            # Decaying Epsilon & Alpha schedules (Slides 45, 48)
            epsilon = max(self.epsilon_min, self.epsilon_initial * ((1.0 - progress) ** 1.5))
            alpha = max(self.alpha_min, self.alpha_initial * (1.0 - 0.7 * progress))

            state = self.env.reset()
            speed, lane, min_dist = state
            state_idx = (int(speed) * 2500 + int(lane) * 625 +
                         int(min_dist[0]) * 125 + int(min_dist[1]) * 25 +
                         int(min_dist[2]) * 5 + int(min_dist[3]))
            done = False
            trajectory = []

            while not done:
                step_counter += 1
                if (step_counter & 63) == 0:
                    if time.time() - start_time >= max_duration:
                        break

                action = self.select_action(state_idx, epsilon)

                next_state, reward, done = self.env.step(action)
                next_speed, next_lane, next_min_dist = next_state
                next_state_idx = (int(next_speed) * 2500 + int(next_lane) * 625 +
                                  int(next_min_dist[0]) * 125 + int(next_min_dist[1]) * 25 +
                                  int(next_min_dist[2]) * 5 + int(next_min_dist[3]))

                # 1. Temporal Difference (TD) Q-Learning Step Update (Slides 28-31, 42-44)
                self.update_q_value(state_idx, action, reward, next_state_idx, done, alpha)
                trajectory.append((state_idx, action, reward, next_state_idx, done))

                state_idx = next_state_idx

            # 2. Episode Trajectory Credit Assignment:
            if len(trajectory) > 0:
                last_reward = trajectory[-1][2]
                last_done = trajectory[-1][4]
                if last_done and last_reward < 0:
                    # Monte Carlo Return G backpropagation for crash episodes (Slides 18-25)
                    G = last_reward
                    for s_idx, act, rwd, n_s_idx, dn in reversed(trajectory[:-1]):
                        G = rwd + df * G
                        self.q_table[s_idx, act] += alpha * (G - self.q_table[s_idx, act])
                else:
                    # Multi-Pass TD Learning for data-efficient credit propagation (Slides 28-31, 42-44)
                    for _pass in range(3):
                        for s_idx, act, rwd, n_s_idx, dn in reversed(trajectory):
                            if dn:
                                tgt = rwd
                            else:
                                tgt = rwd + df * np.max(self.q_table[n_s_idx])
                            self.q_table[s_idx, act] += alpha * (tgt - self.q_table[s_idx, act])

    # ......................... Get Action for Evaluation ..........................
    def get_action(self, speed, lane, min_dist) -> int:
        state_idx = (int(speed) * 2500 + int(lane) * 625 +
                     int(min_dist[0]) * 125 + int(min_dist[1]) * 25 +
                     int(min_dist[2]) * 5 + int(min_dist[3]))
        return self._greedy_action(state_idx)