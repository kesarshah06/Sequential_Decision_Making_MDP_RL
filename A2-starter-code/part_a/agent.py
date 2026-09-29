from env import TreasureHunt
import numpy as np

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

        self._build_ship_transitions()
        self._build_pirate_transitions()

        self.V = np.zeros((4, self.C), dtype=np.float64)

        
        self.policy = np.zeros((4, self.C), dtype=np.int8)

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

            other_actions = [
                a for a in self.ACTIONS
                if a != action
            ]

            outcomes = [action] + other_actions

            for s in range(self.C):

                p = self._pos(s)

                for k, real_action in enumerate(outcomes):

                    di, dj = deltas[real_action]

                    q = (p[0] + di, p[1] + dj)

                    # invalid movement -> remain where we are
                    if not self._valid_ship(q):
                        q = p

                    self.ship_next[action, k, s] = self._idx(q)

    def _build_pirate_transitions(self):
    
        # pirate_next[0] -> Pirate 1
        # pirate_next[1] -> Pirate 2
        #
        # Each position maps to:
        # ((next_position, probability), ...)
        self.pirate_next = [{}, {}]

        deltas = (
            (1, 0),
            (-1, 0),
            (0, -1),
            (0, 1)
        )

        for pirate in (0, 1):

            allowed = set(self.env.pirate_areas[pirate])

            probabilities = self.pirate_prob[pirate]

            table = self.pirate_next[pirate]

            for position in allowed:

                distribution = {}

                for action, probability in enumerate(probabilities):

                    if probability == 0.0:
                        continue

                    q = (position[0] + deltas[action][0],
                        position[1] + deltas[action][1])

                    # Invalid pirate movement -> stay
                    if q not in allowed:
                        q = position

                    distribution[q] = (distribution.get(q, 0.0) + probability)


                table[position] = tuple(distribution.items())

    def _pirate_dist(self, position, pirate):
        return dict(self.pirate_next[pirate][tuple(position)])

    def _mask(self, treasures):

        mask = 0

        for p in treasures:

            bit = self.tresure_to_bit.get(
                tuple(p)
            )

            if bit is not None:
                mask |= bit

        return mask

    def _bellman_action(self, mask, action, values):

        q = np.zeros(self.C, dtype=np.float64)

        for k in range(4):

            probability = self.ship_prob[action, k]

            if probability == 0.0:
                continue

            nxt = self.ship_next[action, k] 
            

            bits = self.tresure_bit[nxt]

            # If we move onto a remaining treasure,
            # remove it from the state.
            next_mask = (mask & (~bits.astype(np.int64))) & 3

            reward = np.full(self.C, self.rewards["step"], dtype=np.float64)

            has_treasure = ((bits & mask) != 0)

            reward[has_treasure] += (self.rewards["treasure"])

            reward[nxt == self.fort_idx] += self.rewards["fort"]

            continuation = values[next_mask, nxt]

            # Fort is terminal
            continuation[nxt == self.fort_idx] = 0.0

            q += probability * (reward + self.gamma * continuation)

        q[self.land_mask] = -np.inf

        return q

    def _compute_base_values(self):
    
        # There are only four possible treasure masks:
        #
        # 00 -> no treasures remain
        # 01 -> treasure 1 remains
        # 10 -> treasure 2 remains
        # 11 -> both remain
        #
        # A transition that collects a treasure moves from a mask
        # to a mask with fewer bits. Therefore we solve masks in
        # increasing number of remaining treasures.

        masks = sorted(range(4), key=lambda x: x.bit_count())

        for mask in masks:

            v = self.V[mask]

            v[:] = 0.0
            v[self.land_mask] = 0.0
            v[self.fort_idx] = 0.0

            for _ in range(1600):

                old = v.copy()

                qs = np.stack(
                    [self._bellman_action(
                            mask,
                            action,
                            self.V)
                        for action in self.ACTIONS
                    ],
                    axis=0
                )

                v[:] = np.max(qs, axis=0)

                v[self.land_mask] = 0.0
                v[self.fort_idx] = 0.0

                if np.max(np.abs(v - old)) < 1e-9:
                    break

            # Extract greedy policy
            qs = np.stack(
                [self._bellman_action(
                        mask,
                        action,
                        self.V)
                    for action in self.ACTIONS],
                axis=0
            )

            self.policy[mask] = (np.argmax(qs, axis=0).astype(np.int8))

    def _future_risk(self, start_idx, mask, p1_dist, p2_dist, horizon=7):

        # Distribution over:
        # (ship_position, treasure_mask)
        ship = {(start_idx, mask): 1.0}

        d1 = p1_dist
        d2 = p2_dist

        exposure = 0.0
        survival = 1.0

        for t in range(1,   horizon + 1):

            # Pirates move first.
            next_d1 = {}
            next_d2 = {}

            for p, probability in d1.items():

                for q, pq in self.pirate_next[0][p]:

                    next_d1[q] = (next_d1.get(q, 0.0)+ probability * pq)

            for p, probability in d2.items():

                for q, pq in self.pirate_next[1][p]:

                    next_d2[q] = (next_d2.get(q, 0.0) + probability * pq)

            d1 = next_d1
            d2 = next_d2

            new_ship = {}
            collision_mass = 0.0

            for (s, m), ship_probability in ship.items():

                action = int(self.policy[m, s])

                for k in range(4):

                    action_probability = (self.ship_prob[action, k])

                    if action_probability == 0.0:
                        continue

                    ns = int(self.ship_next[action, k, s])

                    # Reaching fort terminates successfully.
                    if ns == self.fort_idx:
                        continue

                    position = self._pos(ns)

                    pirate_probability = (d1.get(position, 0.0) + d2.get(position, 0.0))

                    mass = (ship_probability * action_probability)


                    collision_mass += (mass * pirate_probability)

                    safe_mass = (mass * (1.0 - pirate_probability))

                    if safe_mass == 0.0:
                        continue

                    next_mask = (m & (~int(self.tresure_bit[ns])))

                    key = (ns, next_mask)

                    new_ship[key] = (new_ship.get(key, 0.0) + safe_mass)

            exposure += (survival * collision_mass * (self.gamma ** t))

            survival *= max(0.0, 1.0 - collision_mass)

            ship = new_ship

            if not ship:
                break

        return exposure


    
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

        Note:
            This function may be called multiple times after
            `learn_policy()` has been executed.
        """

        if not self.ready:
            self.learn_policy(1.0)

        ship_idx = self._idx(tuple(ship_location))
        mask = self._mask(treasure_locations)

        return int(self.policy[mask, ship_idx])


    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.

        The learned policy should be stored internally by the agent
        and subsequently used by `get_action()`.

        Returns:
            None
        """

        # TODO

        self._compute_base_values()
        
        self.ready = True