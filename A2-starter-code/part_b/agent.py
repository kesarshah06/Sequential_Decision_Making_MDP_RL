import random
import time as time_
import numpy as np

from env import HighwayEnv

ACTION_INC_SPEED = 0
ACTION_DEC_SPEED = 1
ACTION_INC_LANE = 2
ACTION_DEC_LANE = 3
ACTION_NOOP = 4

# action mirror mapping for left-right lane symmetry
FLIP_MAP = (0, 1, 3, 2, 4)


class Agent:
    #..................................................initialize the agent..................................................
    def __init__(self, env: HighwayEnv, discount_factor = 0.99):
        self.env = env                                                                            # highway environment used for training
        self.df = discount_factor                                                                 # discount factor for future rewards
        self.num_actions = 5                                                                      # number of available actions
        self.Q = {}                                                                               # q-values for each encoded state
        self.N = {}                                                                               # visit counts for each state-action pair (online only)

        self.horizon = 4                                                                          # multi-step lookahead horizon
        self.num_iterations = 3                                                                   # number of multi-step iterations per episode
        self.opt_horizon = 25                                                                     # horizon for optimistic q-initialization
        self.eps_start = 0.40                                                                     # initial exploration rate
        self.eps_min = 0.01                                                                       # minimum exploration rate
        self.lr_c = 3.0                                                                           # learning rate constant for harmonic decay
        self.lr_floor = 0.05                                                                      # minimum learning rate floor
        self.tie_frac = 0.005                                                                     # tolerance fraction for greedy tie-breaking
        self.q0 = 0.0                                                                             # optimistic initialization value (set in learn_policy)
        self.tol = 0.0                                                                            # tolerance for greedy tie-breaking (set in learn_policy)

        self._wall = 9                                                                            # sentinel value for unavailable lanes
        self.total_steps = 0                                                                      # total number of training steps

    #..................................................encode the state using discretized sensor inputs..................................................
    def encode_state(self, speed, lane, min_dist):
        """
        encodes state representation capturing:
        - full discretized speed
        - lane position (edge vs interior)
        - distance bins per lane
        - danger flag for close obstacles
        uses left-right symmetry to halve state space.
        """
        lane = int(lane)
        speed = int(speed)

        own_d = int(min_dist[lane])
        left_d = int(min_dist[lane - 1]) if lane > 0 else self._wall
        right_d = int(min_dist[lane + 1]) if lane < 3 else self._wall

        danger = 1 if own_d <= 1 else 0
        l_type = 0 if (lane == 0 or lane == 3) else 1

        if left_d > right_d:
            s_key = (speed, l_type, danger, own_d, right_d, left_d)
            return s_key, True

        s_key = (speed, l_type, danger, own_d, left_d, right_d)
        return s_key, False

    #..................................................get q-values and visit counts for a state..................................................
    def _get_q_values(self, state):
        """retrieves q-values and visit counts, initializing with optimistic q0."""
        if state not in self.Q:                                                                    # initialize unseen states with optimistic values
            self.Q[state] = [self.q0] * 5                                                          # optimistic q-values encourage exploration
            self.N[state] = [0] * 5                                                                # start action visit counts at zero
        return self.Q[state]

    #..................................................select a greedy action from the q-values..................................................
    def _select_greedy_action(self, q_values, state=None):
        """
        greedy action selection with smart tie-breaking:
        - when safe (no danger), prefer increasing speed
        - when in danger, prefer lane change to safer side or decelerate
        """
        max_q = max(q_values)
        tol = self.tol

        candidates = [a for a in range(5) if q_values[a] >= max_q - tol]

        if len(candidates) == 1:
            return candidates[0]

        danger = state[2] if (state is not None and len(state) > 2) else 0

        if danger:
            # in danger: prefer lane changes (escape), then decelerate, then no-op, then accelerate
            priority = [ACTION_INC_LANE, ACTION_DEC_LANE, ACTION_DEC_SPEED, ACTION_NOOP, ACTION_INC_SPEED]
        else:
            # safe: prefer accelerate, then no-op, then lane changes
            priority = [ACTION_INC_SPEED, ACTION_NOOP, ACTION_INC_LANE, ACTION_DEC_LANE, ACTION_DEC_SPEED]

        for act in priority:
            if act in candidates:
                return act
        return candidates[0]

    #..................................................probe the environment to estimate the reward scale..................................................
    def _probe_reward_scale(self):
        """briefly probes environment at startup to calculate optimistic initialization values."""
        probe_env = HighwayEnv()                                                                   # use a separate environment for probing
        probe_env.reset(seed=random.randrange(2 ** 31))                                           # randomize the probe start state
        max_r = 0.0                                                                                # track the largest observed reward
        for _ in range(10):
            _, reward, done = probe_env.step(ACTION_INC_SPEED)
            max_r = max(max_r, reward)
            if done:
                break
        return max_r if max_r > 0 else 0.1

    #..................................................learn a policy with tabular q-learning..................................................
    def learn_policy(self, time):
        t_start = time_.time()
        margin = 0.35 if time >= 5 else 0.03
        max_dur = max(0.05, time - margin)
        gamma = self.df
        horizon = self.horizon

        rmax = self._probe_reward_scale()                                                          # estimate the reward scale for initialization
        if gamma < 1.0:
            self.q0 = rmax * (1.0 - gamma ** self.opt_horizon) / (1.0 - gamma)
            self.tol = self.tie_frac * rmax / (1.0 - gamma)
        else:
            self.q0 = rmax * self.opt_horizon
            self.tol = 0.0

        total_steps = 0
        episodes = 0
        stop = False

        while not stop:
            elapsed = time_.time() - t_start
            if elapsed >= max_dur:
                break

            pct = min(1.0, elapsed / max_dur)
            eps = self.eps_min + 0.5 * (self.eps_start - self.eps_min) * (1.0 + np.cos(np.pi * pct))

            env = HighwayEnv()
            speed, lane, min_dist = env.reset(seed=random.randrange(2 ** 31))
            st, flipped = self.encode_state(speed, lane, min_dist)
            done = False
            traj = []
            episodes += 1

            while not done:
                total_steps += 1
                if (total_steps & 15 == 0) and (time_.time() - t_start >= max_dur):
                    stop = True
                    break

                q_vals = self._get_q_values(st)
                if random.random() < eps:
                    act = random.randrange(5)
                else:
                    act = self._select_greedy_action(q_vals, st)

                env_act = FLIP_MAP[act] if flipped else act
                (next_spd, next_lane, next_dist), reward, done = env.step(env_act)
                nxt_st, nxt_flipped = self.encode_state(next_spd, next_lane, next_dist)

                crash = bool(done and reward < 0)                                                  # identify crash terminations
                nxt_q = self._get_q_values(nxt_st)                                                 # retrieve next-state estimates
                target = reward if crash else reward + gamma * max(nxt_q)                          # compute the temporal-difference target

                counts = self.N[st]                                                                # retrieve visit counts for the current state
                alpha = max(self.lr_floor, self.lr_c / (self.lr_c + counts[act]))                 # use a decaying learning rate
                counts[act] += 1                                                                   # record this action visit
                q_vals[act] += alpha * (target - q_vals[act])                                      # apply the one-step q-learning update

                traj.append((st, act, reward, nxt_st, crash))
                st, flipped = nxt_st, nxt_flipped

            t_len = len(traj)
            if t_len == 0:
                continue

            # multi step with 3 iteration over the trajectory
            for _iter in range(self.num_iterations):
                for t in range(t_len - 1, -1, -1):
                    G = 0.0
                    disc = 1.0
                    is_term = False
                    end_idx = t

                    for k in range(t, min(t + horizon, t_len)):
                        G += disc * traj[k][2]
                        disc *= gamma
                        end_idx = k
                        if traj[k][4]:
                            is_term = True
                            break

                    if not is_term:
                        nq = self._get_q_values(traj[end_idx][3])
                        G += disc * max(nq)

                    s_t, a_t = traj[t][0], traj[t][1]
                    counts_t = self.N[s_t]
                    alpha = max(self.lr_floor, self.lr_c / (self.lr_c + counts_t[a_t]))
                    iter_alpha = alpha * 0.7                                                       # slightly reduced learning rate for trajectory iterations
                    self.Q[s_t][a_t] += iter_alpha * (G - self.Q[s_t][a_t])

        self.total_steps = total_steps

    #..................................................get the best action for the current state..................................................
    def get_action(self, speed, lane, min_dist) -> int:
        st, flipped = self.encode_state(speed, lane, min_dist)
        q_vals = self._get_q_values(st)
        act = self._select_greedy_action(q_vals, st)
        return FLIP_MAP[act] if flipped else act