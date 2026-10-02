from env import TreasureHunt
import numpy as np
import time

class Agent:

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [UP, DOWN, LEFT, RIGHT]

    def __init__(self, layout_file, prob_file):
        """
        Initialize the agent.

        Args:
            layout_file: Path to the grid layout file.
            prob_file: Path to the file containing environment probabilities.

        You may use this function to:
            - Read and store the grid layout.
            - Read and store transition probabilities.
            - Identify important locations such as the fort.
            - Construct the state space and transition model.
            - Initialize any data structures required for learning.
        """

        # TODO

        #.......................................................implementation.................................................#
        self.env = TreasureHunt(layout_file, prob_file)

        self.N = self.env.N
        self.C = self.N * self.N
        self.gamma = self.env.df

        self.ps = self.env.ship_prob[0]
        self.pirate_prob = self.env.pirate_prob
        self.rewards = self.env.rewards

        self.land = set(self.env.locations["land"])
        self.fort = next(iter(self.env.locations["fort"]))

        self.original_tresures = tuple(self.env.locations["treasure"])
        self.tresure_to_bit = {
            p : 1<<i
            for i, p in enumerate(self.original_tresures)
        }

        self.tresure_bit = np.zeros(self.C, dtype=np.int32)

        for p,bit in self.tresure_to_bit.items():
            self.tresure_bit[self._idx(p)] = bit

        self.fort_idx = self._idx(self.fort)

        self.land_mask = np.zeros(self.C, dtype=bool)
        
        for p in self.land:
            self.land_mask[self._idx(p)] = True

        self.p1_area = list(self.env.pirate_areas[0])
        self.p2_area = list(self.env.pirate_areas[1])
        self.n1 = len(self.p1_area)
        self.n2 = len(self.p2_area)
        self.p1_to_idx = {p: i for i, p in enumerate(self.p1_area)}
        self.p2_to_idx = {p: i for i, p in enumerate(self.p2_area)}

        self._build_ship_transitions()
        self._build_pirate_transitions()

        self.V = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)
        self.policy = np.zeros((4, self.n1, self.n2, self.C), dtype=np.int8)

        self.ready = False

    def _idx(self, p):
        return p[0] * self.N + p[1]

    def _pos(self, idx):
        return divmod(int(idx), self.N)

    def _valid_ship(self, p):
        i, j = p
        if i < 0 or i >= self.N:
            return False
        if j < 0 or j >= self.N:
            return False
        if p in self.land:
            return False
        return True

    def _build_ship_transitions(self):
        self.ship_next = np.empty((4, 4, self.C), dtype=np.int32)
        self.ship_prob = np.empty((4, 4), dtype=np.float64)

        deltas = (
            (1, 0),    # UP
            (-1, 0),   # DOWN
            (0, -1),   # LEFT
            (0, 1)     # RIGHT
        )

        for action in self.ACTIONS:
            self.ship_prob[action, 0] = self.ps
            self.ship_prob[action, 1:] = ((1.0 - self.ps) / 3.0)

            other_actions = [a for a in self.ACTIONS if a != action]
            outcomes = [action] + other_actions

            for s in range(self.C):
                p = self._pos(s)
                for k, real_action in enumerate(outcomes):
                    di, dj = deltas[real_action]
                    q = (p[0] + di, p[1] + dj)
                    if not self._valid_ship(q):
                        q = p
                    self.ship_next[action, k, s] = self._idx(q)

    def _build_pirate_transitions(self):
        deltas = (
            (1, 0),
            (-1, 0),
            (0, -1),
            (0, 1)
        )

        self.P1 = np.zeros((self.n1, self.n1), dtype=np.float64)
        for i, pos in enumerate(self.p1_area):
            for action, prob in enumerate(self.pirate_prob[0]):
                if prob == 0.0:
                    continue
                q = (pos[0] + deltas[action][0], pos[1] + deltas[action][1])
                if q not in self.p1_to_idx:
                    q = pos
                self.P1[i, self.p1_to_idx[q]] += prob

        self.P2 = np.zeros((self.n2, self.n2), dtype=np.float64)
        for i, pos in enumerate(self.p2_area):
            for action, prob in enumerate(self.pirate_prob[1]):
                if prob == 0.0:
                    continue
                q = (pos[0] + deltas[action][0], pos[1] + deltas[action][1])
                if q not in self.p2_to_idx:
                    q = pos
                self.P2[i, self.p2_to_idx[q]] += prob

        for i in range(self.n1):
            s = np.sum(self.P1[i])
            if s > 0.0:
                self.P1[i] /= s

        for i in range(self.n2):
            s = np.sum(self.P2[i])
            if s > 0.0:
                self.P2[i] /= s

        # Precompute collision probabilities: col_prob[i1, i2, s_prime]
        self.col_prob = np.zeros((self.n1, self.n2, self.C), dtype=np.float64)
        for i1, p1 in enumerate(self.p1_area):
            for s_prime in self.p1_area:
                s_idx = self._idx(s_prime)
                self.col_prob[i1, :, s_idx] = self.P1[i1, self.p1_to_idx[s_prime]]
        for i2, p2 in enumerate(self.p2_area):
            for s_prime in self.p2_area:
                s_idx = self._idx(s_prime)
                self.col_prob[:, i2, s_idx] = self.P2[i2, self.p2_to_idx[s_prime]]

        # Precompute immediate expected reward: ImmR[mask, i1, i2, s_prime]
        self.ImmR = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)
        for m in range(4):
            self.ImmR[m] = self.rewards["step"] + self.col_prob * self.rewards["pirate"]
            self.ImmR[m, :, :, self.fort_idx] += self.rewards["fort"]
            for p, bit in self.tresure_to_bit.items():
                if m & bit:
                    t_idx = self._idx(p)
                    self.ImmR[m, :, :, t_idx] += self.rewards["treasure"]

    def _mask(self, treasures):
        mask = 0
        for p in treasures:
            bit = self.tresure_to_bit.get(tuple(p))
            if bit is not None:
                mask |= bit
        return mask

    def _compute_full_values(self, time_budget=1.0):
        t_start = time.time()
        deadline = t_start + max(0.01, 0.85 * float(time_budget))

        masks = sorted(range(4), key=lambda x: x.bit_count())
        W = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)

        for m_idx, mask in enumerate(masks):
            rem_masks = len(masks) - m_idx
            mask_deadline = time.time() + (deadline - time.time()) / rem_masks
            rem_treasures = [p for p, bit in self.tresure_to_bit.items() if mask & bit]

            for _ in range(500):
                W[mask] = np.einsum('ia,jb,abs->ijs', self.P1, self.P2, self.V[mask])
                Cont = self.gamma * W[mask].copy()

                for p in rem_treasures:
                    t_idx = self._idx(p)
                    next_mask = mask ^ self.tresure_to_bit[p]
                    Cont[:, :, t_idx] = self.gamma * W[next_mask, :, :, t_idx]

                Cont[:, :, self.fort_idx] = 0.0

                Target = self.ImmR[mask] + Cont

                Q = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)
                for a in self.ACTIONS:
                    for k in range(4):
                        prob = self.ship_prob[a, k]
                        if prob > 0.0:
                            Q[a] += prob * Target[:, :, self.ship_next[a, k]]

                new_v = np.max(Q, axis=0)

                # Terminal states in V
                new_v[:, :, self.land_mask] = 0.0
                new_v[:, :, self.fort_idx] = 0.0
                for i1, p1 in enumerate(self.p1_area):
                    new_v[i1, :, self._idx(p1)] = 0.0
                for i2, p2 in enumerate(self.p2_area):
                    new_v[:, i2, self._idx(p2)] = 0.0

                diff = np.max(np.abs(new_v - self.V[mask]))
                self.V[mask] = new_v
                self.policy[mask] = np.argmax(Q, axis=0).astype(np.int8)

                if diff < 1e-4 or time.time() >= mask_deadline:
                    break

            # Update continuation matrix W[mask]
            W[mask] = np.einsum('ia,jb,abs->ijs', self.P1, self.P2, self.V[mask])

    def get_action(self, ship_location, pirate_locations, treasure_locations) -> int:
        """
        Choose an action for the current state.

        Args:
            ship_location: Tuple (x, y) representing the current
                location of the ship.
            pirate_locations: List containing the locations of all
                pirates. There can be at most 2 pirates.
            treasure_locations: List containing the locations of all
                treasures. There can be at most 2 treasures.

        Returns:
            An integer representing the selected action:
                0 -> UP
                1 -> DOWN
                2 -> LEFT
                3 -> RIGHT
        """
        if not self.ready:
            self.learn_policy(1.0)

        ship_idx = self._idx(tuple(ship_location))
        mask = self._mask(treasure_locations)

        p1_loc = tuple(pirate_locations[0])
        p2_loc = tuple(pirate_locations[1])

        if p1_loc in self.p1_to_idx:
            p1_i = self.p1_to_idx[p1_loc]
            p2_i = self.p2_to_idx.get(p2_loc, 0)
        else:
            p1_i = self.p1_to_idx.get(p2_loc, 0)
            p2_i = self.p2_to_idx.get(p1_loc, 0)

        return int(self.policy[mask, p1_i, p2_i, ship_idx])

    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.
        """
        self._compute_full_values(time)
        self.ready = True