from env import TreasureHunt
import numpy as np
import time

class Agent:

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTIONS = [UP, DOWN, LEFT, RIGHT]

#...................................................................................init.................................................................................#
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

        self.env = TreasureHunt(layout_file, prob_file)

        self.N = self.env.N                                                                            # N = grid size
        self.C = self.N * self.N                                                                       # C = number of cells in the grid
        self.gamma = self.env.df                                                                       # discount factor

        self.ps = self.env.ship_prob[0]                                                                # probability of ship moving in the intended direction
        self.pirate_prob = self.env.pirate_prob                                                        # probability of pirate moving in the intended direction
        self.rewards = self.env.rewards                                                                # reward structure for the environment

        self.land = set(self.env.locations["land"])                                                    # Set of land locations in the grid
        self.fort = next(iter(self.env.locations["fort"]))                                             # The fort location in the grid

        self.original_tresures = tuple(self.env.locations["treasure"])                                 # The original treasure locations in the grid
        self.tresure_to_bit = {
            p : 1<<i
            for i, p in enumerate(self.original_tresures)
        }                                                                                              # p = treasure location, i = index of the treasure in the original_tresures list, 1<<i = bit representation for state encoding 

        self.tresure_bit = np.zeros(self.C, dtype=np.int32)                                            # Mapping from cell index to bit representation for state encoding

        for p,bit in self.tresure_to_bit.items():
            self.tresure_bit[self.idx(p)] = bit                                                       

        self.fortidx = self.idx(self.fort)                                                             # Index of the fort location in the grid

        self.land_mask = np.zeros(self.C, dtype=bool)                                                  # Mask indicating land locations in the grid
        
        for p in self.land:                                                                            # set the land_mask to True for all land locations in the grid
            self.land_mask[self.idx(p)] = True

        self.p1_area = list(self.env.pirate_areas[0])                                                  # List of locations for pirate 1 in the grid
        self.p2_area = list(self.env.pirate_areas[1])                                                  # List of locations for pirate 2 in the grid
        self.n1 = len(self.p1_area)                                                                    # Number of locations for pirate 1 in the grid
        self.n2 = len(self.p2_area)                                                                    # Number of locations for pirate 2 in the grid
        self.p1_toidx = {p: i for i, p in enumerate(self.p1_area)}                                     # Mapping from pirate 1 location to index in the p1_area list
        self.p2_toidx = {p: i for i, p in enumerate(self.p2_area)}                                     # Mapping from pirate 2 location to index in the p2_area list

        self.build_ship_transitions()                                                                  # Build the transition model for the ship's movement in the grid
        self.build_pirate_transitions()                                                                # Build the transition model for the pirates' movement in the grid

        self.V = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)                             # Value function for each state in the grid, indexed by the treasure mask, pirate 1 index, pirate 2 index, and ship index
        self.policy = np.zeros((4, self.n1, self.n2, self.C), dtype=np.int8)                           # Policy for each state in the grid, indexed by the treasure mask, pirate 1 index, pirate 2 index, and ship index

        self.ready = False                                                                             # Flag indicating whether the agent has learned a policy for navigating the Treasure Hunt environment

#....................................function for state encoding,(rather than using state as a pair (x,y) we use a single integer to represent the state)..................................................
    def idx(self, p):
        return p[0] * self.N + p[1]

#...............................................................function for state decoding..................................................................................
    def pos(self, idx):
        return divmod(int(idx), self.N)

#............................................................function to check if a ship position is valid..................................................................................
    def valid_ship(self, p):
        i, j = p
        if i < 0 or i >= self.N:
            return False
        if j < 0 or j >= self.N:
            return False
        if p in self.land:
            return False
        return True

#...........................................................function to build the ship's transition model..................................................................................
    def build_ship_transitions(self):
        self.ship_next = np.empty((4, 4, self.C), dtype=np.int32)                                     # empty array to store the next state index for each action, outcome, and current state index            
        self.ship_prob = np.empty((4, 4), dtype=np.float64)                                           # empty array to store the transition probabilities for each action and outcome

        deltas = (
            (1, 0),                                                                                   # UP
            (-1, 0),                                                                                  # DOWN
            (0, -1),                                                                                  # LEFT
            (0, 1)                                                                                    # RIGHT
        )

        for action in self.ACTIONS:
            self.ship_prob[action, 0] = self.ps                                                       # probability of moving in the intended direction
            self.ship_prob[action, 1:] = ((1.0 - self.ps) / 3.0)                                      # probability of moving in the other three directions

            other_actions = [a for a in self.ACTIONS if a != action]                          
            outcomes = [action] + other_actions                                                       # making action our first outcome and the other three actions as the other outcomes

            for s in range(self.C):                                                           
                p = self.pos(s)                                                                       # get the (x,y) position of the ship for the current state index
                for k, real_action in enumerate(outcomes):                           
                    di, dj = deltas[real_action]
                    q = (p[0] + di, p[1] + dj)                                                        # get the (x,y) position of the ship for the next state index based on the action taken
                    if not self.valid_ship(q):                                                        # if the next position is not valid, stay in the same position
                        q = p
                    self.ship_next[action, k, s] = self.idx(q)                                        # get the index of the next state based on the (x,y) position of the ship for the next state index

#...........................................................function to build the pirates' transition model..................................................................................
    def build_pirate_transitions(self):
        deltas = (
            (1, 0),                                                                                   # UP
            (-1, 0),                                                                                  # DOWN
            (0, -1),                                                                                  # LEFT
            (0, 1)                                                                                    # RIGHT
        )

        self.P1 = np.zeros((self.n1, self.n1), dtype=np.float64)                                      # Transition matrix for pirate 1's movement, where P1[i, j] represents the probability of moving from location i to location j
        for i, pos in enumerate(self.p1_area):
            for action, prob in enumerate(self.pirate_prob[0]):                                       # For each action and its corresponding probability for pirate 1's movement
                if prob == 0.0:
                    continue
                q = (pos[0] + deltas[action][0], pos[1] + deltas[action][1])                          # Get the next position of pirate 1 based on the action taken
                if q not in self.p1_toidx:                                                            # If the next position is not valid, stay in the same position
                    q = pos
                self.P1[i, self.p1_toidx[q]] += prob                                                  # Update the transition probability for moving from location i to location j based on the action taken and its corresponding probability

        self.P2 = np.zeros((self.n2, self.n2), dtype=np.float64)                                      # Transition matrix for pirate 2's movement, where P2[i, j] represents the probability of moving from location i to location j
        for i, pos in enumerate(self.p2_area):
            for action, prob in enumerate(self.pirate_prob[1]):
                if prob == 0.0:
                    continue
                q = (pos[0] + deltas[action][0], pos[1] + deltas[action][1])                           # Get the next position of pirate 2 based on the action taken
                if q not in self.p2_toidx:                                                             # If the next position is not valid, stay in the same position
                    q = pos
                self.P2[i, self.p2_toidx[q]] += prob                                                   # Update the transition probability for moving from location i to location j based on the action taken and its corresponding probability

        for i in range(self.n1):                                                                       # Normalize the transition probabilities for pirate 1's movement so that they sum to 1
            s = np.sum(self.P1[i])
            if s > 0.0:
                self.P1[i] /= s

        for i in range(self.n2):                                                                       # Normalize the transition probabilities for pirate 2's movement so that they sum to 1
            s = np.sum(self.P2[i])
            if s > 0.0:
                self.P2[i] /= s

        self.col_prob = np.zeros((self.n1, self.n2, self.C), dtype=np.float64)                         # Collision probabilities between pirates and the environment
        for i1, p1 in enumerate(self.p1_area):                                                         # For each location of pirate 1, compute the collision probabilities with the environment
            for s_prime in self.p1_area:
                sidx = self.idx(s_prime)
                self.col_prob[i1, :, sidx] = self.P1[i1, self.p1_toidx[s_prime]]
        for i2, p2 in enumerate(self.p2_area):                                                         # For each location of pirate 2, compute the collision probabilities with the environment
            for s_prime in self.p2_area:
                sidx = self.idx(s_prime)
                self.col_prob[:, i2, sidx] = self.P2[i2, self.p2_toidx[s_prime]]

        self.ImmR = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)                         
        for m in range(4):
            self.ImmR[m] = self.rewards["step"] + self.col_prob * self.rewards["pirate"]               # Immediate rewards for each state in the grid, indexed by the treasure mask, pirate 1 index, pirate 2 index, and ship index
            self.ImmR[m, :, :, self.fortidx] += self.rewards["fort"]                        
            for p, bit in self.tresure_to_bit.items():
                if m & bit:
                    tidx = self.idx(p)
                    self.ImmR[m, :, :, tidx] += self.rewards["treasure"]

#.........................................function to compute the treasure mask for a given set of treasures..................................................................................
    def _mask(self, treasures):                                                                   
        mask = 0
        for p in treasures:         
            bit = self.tresure_to_bit.get(tuple(p))
            if bit is not None:
                mask |= bit
        return mask

#..................................................function to get the Q-values for a given state..................................................................................
    def compute_full_values(self, time_budget=1.0):
        t_start = time.time()                                                                          # Start time for the value iteration process
        deadline = t_start + max(0.01, 0.85 * float(time_budget))                                      # End time for the value iteration process, set to 85% of the time budget or at least 0.01 seconds

        masks = sorted(range(4), key=lambda x: x.bit_count())                                          # Sort the treasure masks based on the number of treasures present in the mask, so that we can compute the values for states with fewer treasures first
        W = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)                                  # Initialize the continuation matrix W, which will store the expected values for each state in the grid, indexed by the treasure mask, pirate 1 index, pirate 2 index, and ship index

        for midx, mask in enumerate(masks):
            rem_masks = len(masks) - midx
            mask_deadline = time.time() + (deadline - time.time()) / rem_masks
            rem_treasures = [p for p, bit in self.tresure_to_bit.items() if mask & bit]

            for _ in range(500):                                                                       # Maximum number of iterations for value iteration
                W[mask] = np.einsum('ia,jb,abs->ijs', self.P1, self.P2, self.V[mask])                  # Compute the expected values for each state in the grid based on the transition probabilities of the pirates and the current value function V
                Cont = self.gamma * W[mask].copy()

                for p in rem_treasures:                                                                # For each remaining treasure, compute the expected values for the states where the treasure is present, and update the continuation matrix Cont accordingly
                    tidx = self.idx(p)
                    next_mask = mask ^ self.tresure_to_bit[p]
                    Cont[:, :, tidx] = self.gamma * W[next_mask, :, :, tidx]

                Cont[:, :, self.fortidx] = 0.0                                                         # Set the expected values for the states where the ship is at the fort to 0, since the game ends when the ship reaches the fort

                Target = self.ImmR[mask] + Cont                                                        # Compute the target values for each state in the grid based on the immediate rewards and the expected values from the continuation matrix Cont

                Q = np.zeros((4, self.n1, self.n2, self.C), dtype=np.float64)
                for a in self.ACTIONS:                                                                 # For each action, compute the expected Q-values for each state in the grid based on the transition probabilities of the ship and the target values from the previous step
                    for k in range(4):
                        prob = self.ship_prob[a, k]
                        if prob > 0.0:
                            Q[a] += prob * Target[:, :, self.ship_next[a, k]]                          # Update the Q-values for each action based on the transition probabilities of the ship and the target values from the previous step

                new_v = np.max(Q, axis=0)                                                              # Compute the new value function for each state in the grid by taking the maximum Q-value across all actions

                new_v[:, :, self.land_mask] = 0.0
                new_v[:, :, self.fortidx] = 0.0
                for i1, p1 in enumerate(self.p1_area):                                                 # Set the value function to 0 for all states where pirate 1 is at a land location or at the fort, since these states are invalid
                    new_v[i1, :, self.idx(p1)] = 0.0
                for i2, p2 in enumerate(self.p2_area):                                                 # Set the value function to 0 for all states where pirate 2 is at a land location or at the fort, since these states are invalid
                    new_v[:, i2, self.idx(p2)] = 0.0

                diff = np.max(np.abs(new_v - self.V[mask]))
                self.V[mask] = new_v
                self.policy[mask] = np.argmax(Q, axis=0).astype(np.int8)

                if diff < 1e-4 or time.time() >= mask_deadline:                                        # If the maximum change in the value function is less than a threshold or if the time budget for this mask has been exceeded, break out of the loop and move on to the next mask
                    break

            W[mask] = np.einsum('ia,jb,abs->ijs', self.P1, self.P2, self.V[mask])

#..................................................function to get the action for a given state..................................................................................
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
        if not self.ready:                                                                          # If the agent has not learned a policy yet, learn a policy using the available time budget of 1.0 seconds
            self.learn_policy(1.0)

        shipidx = self.idx(tuple(ship_location))                                                    # Get the index of the ship's current location in the grid
        mask = self._mask(treasure_locations)                                                       # Get the treasure mask for the current set of treasures in the grid

        p1_loc = tuple(pirate_locations[0])
        p2_loc = tuple(pirate_locations[1])

        if p1_loc in self.p1_toidx:                                                                 # If pirate 1's location is in the list of valid locations for pirate 1, get the indices for both pirates based on their current locations
            p1_i = self.p1_toidx[p1_loc]
            p2_i = self.p2_toidx.get(p2_loc, 0)
        else:                                                                                       # If pirate 1's location is not in the list of valid locations for pirate 1, get the indices for both pirates based on their current locations
            p1_i = self.p1_toidx.get(p2_loc, 0)
            p2_i = self.p2_toidx.get(p1_loc, 0)

        return int(self.policy[mask, p1_i, p2_i, shipidx])

#..................................................function to learn a policy for navigating the Treasure Hunt environment..................................................................................
    def learn_policy(self, time):
        """
        Learn a policy for navigating the Treasure Hunt environment.

        Args:
            time: Maximum time (in seconds) allowed for learning.
        """
        self.compute_full_values(time)
        self.ready = True