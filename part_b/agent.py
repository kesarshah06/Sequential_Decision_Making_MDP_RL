import random
import time
import numpy as np

from env import HighwayEnv

ACTION_INCREASE_SPEED = 0
ACTION_DECREASE_SPEED = 1
ACTION_INCREASE_LANE = 2
ACTION_DECREASE_LANE = 3
ACTION_NO_OP = 4

# Action mirror mapping to exploit left-right lane symmetry
FLIP_ACTION = (0, 1, 3, 2, 4)


class Agent:
    #..................................................initialize the agent..................................................
    def __init__(self, env: HighwayEnv, discount_factor=0.99):
        self.env = env                                                                            # Highway environment used for training
        self.discount_factor = discount_factor                                                      # Discount factor for future rewards
        self.num_actions = 5                                                                        # Number of available actions
        self.Q = {}                                                                                 # Q-values for each encoded state
        self.N = {}                                                                                 # Visit counts for each state-action pair
        
        self.n_steps = 3                                                                          # Number of steps used in multi-step updates
        self.replay_ratio = 3
        self.replay_cap = 200000
        self.opt_horizon = 20
        self.eps_initial = 0.5
        self.eps_min = 0.02
        self.lr_c = 2.0
        self.lr_min = 0.1
        self.tie_frac = 0.01
        self.q0 = 0.0
        self.tol = 0.0
        
        self._own_map = (3, 1, 2, 3, 3)                                                           # Mapping for obstacles in the current lane
        self._adj_map = (3, 1, 1, 3, 3)                                                           # Mapping for obstacles in adjacent lanes
        self._wall = 9                                                                             # Sentinel value for unavailable lanes
        self.total_steps = 0                                                                       # Total number of training steps

    #..................................................encode the state using discretized sensor inputs..................................................
    def encode_state(self, speed, lane, min_dist):
        """
        Discretizes sensor inputs and orders adjacent lane obstacle distances 
        consistently to exploit horizontal left-right lane symmetry.
        """
        lane = int(lane)                                                                           # Ensure the lane index is an integer
        own = self._own_map[int(min_dist[lane])]                                                    # Aggregate distance in the current lane
        left = self._adj_map[int(min_dist[lane - 1])] if lane > 0 else self._wall                 # Aggregate distance in the left lane
        right = self._adj_map[int(min_dist[lane + 1])] if lane < 3 else self._wall                # Aggregate distance in the right lane
        speed_flag = 1 if int(speed) >= 3 else 0                                                   # Encode high-speed status

        if left > right:                                                                           # Canonicalize state when the left side has more space
            state_key = (speed_flag * 10 + own) * 100 + right * 10 + left
            return state_key, True
        
        state_key = (speed_flag * 10 + own) * 100 + left * 10 + right
        return state_key, False

    #..................................................get Q-values and visit counts for a state..................................................
    def _get_q_values(self, state):
        """Retrieves Q-values and visit counts, initializing with optimistic q0."""
        if state not in self.Q:                                                                    # Initialize unseen states with optimistic values
            self.Q[state] = [self.q0] * 5                                                          # Optimistic Q-values encourage exploration
            self.N[state] = [0] * 5                                                                # Start action visit counts at zero
        return self.Q[state]

    #..................................................select a greedy action from the Q-values..................................................
    def _select_greedy_action(self, q_values):
        """Greedy action selection with tie-breaking preference for maintaining high speed."""
        max_q = max(q_values)                                                                       # Find the best estimated action value
        if q_values[ACTION_INCREASE_SPEED] >= max_q - self.tol:                                    # Prefer increasing speed when nearly tied
            return ACTION_INCREASE_SPEED
            
        for action in (ACTION_NO_OP, ACTION_DECREASE_SPEED, ACTION_INCREASE_LANE, ACTION_DECREASE_LANE):
            if q_values[action] == max_q:
                return action
        return ACTION_NO_OP

    #..................................................probe the environment to estimate the reward scale..................................................
    def _probe_reward_scale(self):
        """Briefly probes environment at startup to calculate optimistic initialization values."""
        probe_env = HighwayEnv()                                                                   # Use a separate environment for probing
        probe_env.reset(seed=random.randrange(2 ** 31))                                           # Randomize the probe start state
        max_reward = 0.0                                                                           # Track the largest observed reward
        for _ in range(6):
            _, reward, done = probe_env.step(ACTION_INCREASE_SPEED)
            max_reward = max(max_reward, reward)
            if done:
                break
        return max_reward if max_reward > 0 else 0.1

    #..................................................learn a policy with tabular Q-learning..................................................
    def learn_policy(self, time_limit: float):
        start_time = time.time()
        margin = 0.35 if time_limit >= 5 else 0.03
        max_duration = max(0.05, time_limit - margin)
        gamma = self.discount_factor
        n_steps = self.n_steps

        rmax = self._probe_reward_scale()                                                          # Estimate the reward scale for initialization
        if gamma < 1.0:
            self.q0 = rmax * (1.0 - gamma ** self.opt_horizon) / (1.0 - gamma)
            self.tol = self.tie_frac * rmax / (1.0 - gamma)
        else:
            self.q0 = rmax * self.opt_horizon
            self.tol = 0.0

        replay_buffer = []
        buffer_pos = 0
        steps = 0
        stop_training = False

        while not stop_training:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                break
                
            epsilon = max(self.eps_min, self.eps_initial * (1.0 - elapsed / max_duration))       # Reduce exploration as training time elapses

            env = HighwayEnv()
            speed, lane, min_dist = env.reset(seed=random.randrange(2 ** 31))
            state, is_flipped = self.encode_state(speed, lane, min_dist)
            done = False
            trajectory = []

            while not done:
                steps += 1
                if (steps % 16 == 0) and (time.time() - start_time >= max_duration):
                    stop_training = True
                    break

                q_vals = self._get_q_values(state)
                if random.random() < epsilon:
                    action = random.randrange(5)
                else:
                    action = self._select_greedy_action(q_vals)

                env_action = FLIP_ACTION[action] if is_flipped else action
                (next_speed, next_lane, next_min_dist), reward, done = env.step(env_action)
                next_state, next_is_flipped = self.encode_state(next_speed, next_lane, next_min_dist)
                
                is_terminal = bool(done and reward < 0)                                           # Identify crash terminations
                next_q_vals = self._get_q_values(next_state)                                      # Retrieve next-state estimates
                target = reward if is_terminal else reward + gamma * max(next_q_vals)              # Compute the temporal-difference target
                
                counts = self.N[state]                                                            # Retrieve visit counts for the current state
                alpha = max(self.lr_min, self.lr_c / (self.lr_c + counts[action]))                # Use a decaying learning rate
                counts[action] += 1                                                               # Record this action visit
                q_vals[action] += alpha * (target - q_vals[action])                               # Apply the one-step Q-learning update

                trajectory.append((state, action, reward, next_state, is_terminal))
                state, is_flipped = next_state, next_is_flipped

            traj_len = len(trajectory)
            if traj_len == 0:
                continue

            for _ in range(2):
                for t in range(traj_len - 1, -1, -1):
                    G = 0.0
                    discount = 1.0
                    terminal_found = False
                    last_idx = t

                    for k in range(t, min(t + n_steps, traj_len)):
                        G += discount * trajectory[k][2]
                        discount *= gamma
                        last_idx = k
                        if trajectory[k][4]:
                            terminal_found = True
                            break

                    if not terminal_found:
                        next_state_q = self._get_q_values(trajectory[last_idx][3])
                        G += discount * max(next_state_q)

                    s_t, a_t = trajectory[t][0], trajectory[t][1]
                    counts = self.N[s_t]
                    alpha = max(self.lr_min, self.lr_c / (self.lr_c + counts[a_t]))
                    counts[a_t] += 1
                    self.Q[s_t][a_t] += alpha * (G - self.Q[s_t][a_t])

            for transition in trajectory:
                if len(replay_buffer) < self.replay_cap:
                    replay_buffer.append(transition)
                else:
                    replay_buffer[buffer_pos] = transition
                    buffer_pos = (buffer_pos + 1) % self.replay_cap

            buf_size = len(replay_buffer)
            num_replays = int(self.replay_ratio * traj_len)
            for _ in range(num_replays):
                s_i, a_i, r_i, ns_i, term_i = replay_buffer[random.randrange(buf_size)]
                next_q = self._get_q_values(ns_i)
                target_i = r_i if term_i else r_i + gamma * max(next_q)
                
                counts = self.N[s_i]
                alpha = max(self.lr_min, self.lr_c / (self.lr_c + counts[a_i]))
                counts[a_i] += 1
                self.Q[s_i][a_i] += alpha * (target_i - self.Q[s_i][a_i])

        self.total_steps = steps

    #..................................................get the best action for the current state..................................................
    def get_action(self, speed, lane, min_dist) -> int:
        state, is_flipped = self.encode_state(speed, lane, min_dist)
        q_values = self._get_q_values(state)
        action = self._select_greedy_action(q_values)
        return FLIP_ACTION[action] if is_flipped else action
    