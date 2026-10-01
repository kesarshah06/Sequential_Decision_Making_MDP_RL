import os
import random
import shutil
from collections import deque
from env import TreasureHunt

def generate_connected_area(start, size, min_r, max_r, min_c, max_c, occupied_cells):
    """Generate a 4-connected component within bounded box, avoiding occupied cells."""
    area = [start]
    occupied_cells.add(start)
    
    attempts = 0
    while len(area) < size and attempts < 100:
        attempts += 1
        curr = random.choice(area)
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        nr, nc = curr[0] + dr, curr[1] + dc
        if min_r <= nr <= max_r and min_c <= nc <= max_c:
            if (nr, nc) not in occupied_cells:
                area.append((nr, nc))
                occupied_cells.add((nr, nc))
                
    return area

def has_adjacency(area1, area2):
    """Check if any cell in area1 is 4-way adjacent to any cell in area2."""
    set2 = set(area2)
    for r, c in area1:
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (r + dr, c + dc) in set2:
                return True
    return False

def generate_single_test_case(out_dir, N, case_idx, seed=None):
    if seed is not None:
        random.seed(seed)
        
    for attempt in range(200):
        grid = [['W' for _ in range(N)] for _ in range(N)]
        
        # Determine pirate area topology
        # Split grid into two non-overlapping regions for pirate 1 and pirate 2
        # e.g., top half / bottom half, or left half / right half, or diagonal
        split_type = case_idx % 4
        
        max_area_size = min(6, max(1, N // 2))
        s1 = random.randint(1, max_area_size)
        s2 = random.randint(1, max_area_size)
        
        if split_type == 0:  # Top / Bottom
            box1 = (0, N // 2 - 2, 0, N - 1)
            box2 = (N // 2 + 1, N - 1, 0, N - 1)
        elif split_type == 1:  # Left / Right
            box1 = (0, N - 1, 0, N // 2 - 2)
            box2 = (0, N - 1, N // 2 + 1, N - 1)
        elif split_type == 2:  # Quadrant Top-Left / Bottom-Right
            box1 = (0, N // 2 - 1, 0, N // 2 - 1)
            box2 = (N // 2 + 1, N - 1, N // 2 + 1, N - 1)
        else:  # Quadrant Bottom-Left / Top-Right
            box1 = (N // 2 + 1, N - 1, 0, N // 2 - 1)
            box2 = (0, N // 2 - 1, N // 2 + 1, N - 1)
            
        # Check box validity
        if box1[0] > box1[1] or box1[2] > box1[3] or box2[0] > box2[1] or box2[2] > box2[3]:
            # fallback to top/bottom with 1-cell buffer
            box1 = (0, max(0, N // 2 - 1), 0, N - 1)
            box2 = (min(N - 1, N // 2 + 1), N - 1, 0, N - 1)
            
        occupied = set()
        
        # Start for pirate 1
        p1_start = (random.randint(box1[0], box1[1]), random.randint(box1[2], box1[3]))
        area1 = generate_connected_area(p1_start, s1, box1[0], box1[1], box1[2], box1[3], occupied)
        
        # Start for pirate 2
        p2_start = (random.randint(box2[0], box2[1]), random.randint(box2[2], box2[3]))
        area2 = generate_connected_area(p2_start, s2, box2[0], box2[1], box2[2], box2[3], occupied)
        
        if has_adjacency(area1, area2):
            continue
            
        # Place pirate markers: 1 for pirate 1, 2 for pirate 2, ! for other area cells
        grid[area1[0][0]][area1[0][1]] = '1'
        for p in area1[1:]:
            grid[p[0]][p[1]] = '!'
            
        grid[area2[0][0]][area2[0][1]] = '2'
        for p in area2[1:]:
            grid[p[0]][p[1]] = '!'
            
        pirate_cells = set(area1) | set(area2)
        
        # Place Ship (S), Fort (F), Treasures (T1, T2)
        water_candidates = [(r, c) for r in range(N) for c in range(N) if (r, c) not in pirate_cells]
        if len(water_candidates) < 4:
            continue
            
        specials = random.sample(water_candidates, 4)
        S, F, T1, T2 = specials[0], specials[1], specials[2], specials[3]
        grid[S[0]][S[1]] = 'S'
        grid[F[0]][F[1]] = 'F'
        grid[T1[0]][T1[1]] = 'T'
        grid[T2[0]][T2[1]] = 'T'
        
        reserved = pirate_cells | {S, F, T1, T2}
        
        # Add land (L) with diverse topology
        available_for_land = [(r, c) for r in range(N) for c in range(N) if (r, c) not in reserved]
        
        # Choose land density style:
        # 0: Open water (0-5%)
        # 1: Light islands (5-15%)
        # 2: Moderate obstacles (15-25%)
        # 3: Chokepoint/barrier walls
        land_style = case_idx % 4
        if land_style == 0:
            land_ratio = random.uniform(0.0, 0.05)
            num_land = int(len(available_for_land) * land_ratio)
            land_cells = random.sample(available_for_land, num_land)
        elif land_style == 1:
            land_ratio = random.uniform(0.05, 0.15)
            num_land = int(len(available_for_land) * land_ratio)
            land_cells = random.sample(available_for_land, num_land)
        elif land_style == 2:
            land_ratio = random.uniform(0.15, 0.25)
            num_land = int(len(available_for_land) * land_ratio)
            land_cells = random.sample(available_for_land, num_land)
        else:
            # Wall barrier with gaps
            wall_row = N // 2
            land_cells = []
            for c in range(N):
                if (wall_row, c) in available_for_land and random.random() < 0.7:
                    land_cells.append((wall_row, c))
                    
        for r, c in land_cells:
            grid[r][c] = 'L'
            
        # Verify reachability of F, T1, T2 from S via non-land cells
        traversable = {(r, c) for r in range(N) for c in range(N) if grid[r][c] != 'L'}
        visited = set()
        queue = deque([S])
        visited.add(S)
        while queue:
            curr = queue.popleft()
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if (nr, nc) in traversable and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    
        if F not in visited or T1 not in visited or T2 not in visited:
            continue
            
        # Generate prob.txt
        # ps: ship execution probability
        ps_choices = [0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
        ps = round(random.choice(ps_choices), 2)
        
        # Pirate movement distribution
        p_modes = [
            [0.25, 0.25, 0.25, 0.25],       # uniform random
            [0.50, 0.50, 0.00, 0.00],       # vertical patrol
            [0.00, 0.00, 0.50, 0.50],       # horizontal patrol
            [0.40, 0.40, 0.10, 0.10],       # vertical biased
            [0.10, 0.10, 0.40, 0.40],       # horizontal biased
            [0.60, 0.20, 0.10, 0.10],       # upward biased
            [0.20, 0.60, 0.10, 0.10],       # downward biased
            [0.10, 0.10, 0.60, 0.20],       # leftward biased
            [0.10, 0.10, 0.20, 0.60],       # rightward biased
            [0.70, 0.30, 0.00, 0.00],       # biased vertical
        ]
        p1_prob = random.choice(p_modes)
        p2_prob = random.choice(p_modes)
        
        step_r = random.choice([-0.01, -0.02, -0.05, -0.10])
        t_r = random.choice([2.0, 3.0, 4.0, 5.0, 8.0])
        f_r = random.choice([3.0, 5.0, 7.0, 10.0, 15.0])
        p_r = random.choice([-1.0, -2.0, -3.0, -5.0, -10.0])
        df = random.choice([0.90, 0.95, 0.98, 0.99])
        
        os.makedirs(out_dir, exist_ok=True)
        layout_path = os.path.join(out_dir, 'layout.txt')
        prob_path = os.path.join(out_dir, 'prob.txt')
        
        with open(layout_path, 'w') as f:
            for row in grid:
                f.write(''.join(row) + '\n')
                
        with open(prob_path, 'w') as f:
            f.write(f'{ps}\n')
            f.write(f'{p1_prob[0]} {p1_prob[1]} {p1_prob[2]} {p1_prob[3]}\n')
            f.write(f'{p2_prob[0]} {p2_prob[1]} {p2_prob[2]} {p2_prob[3]}\n')
            f.write(f'{step_r} {t_r} {f_r} {p_r}\n')
            f.write(f'{df}\n')
            
        # Verify with TreasureHunt environment
        try:
            env = TreasureHunt(layout_path, prob_path)
            # Ensure pirate areas are disjoint
            assert len(env.pirate_areas) == 2
            assert len(env.pirate_areas[0]) >= 1 and len(env.pirate_areas[1]) >= 1
            return True
        except Exception:
            continue
            
    return False

def generate_1000_tests(base_dir='tests', start_id=4, total_count=1000):
    print(f"Generating {total_count} diverse, verified test cases in {base_dir}/...")
    
    # Grid size schedule:
    # 250 small (N = 5..10)
    # 450 medium (N = 11..20)
    # 300 large (N = 21..30)
    sizes = []
    for i in range(total_count):
        if i < 250:
            sizes.append(random.choice([5, 6, 7, 8, 9, 10]))
        elif i < 700:
            sizes.append(random.choice([11, 12, 14, 15, 16, 18, 20]))
        else:
            sizes.append(random.choice([22, 24, 25, 26, 28, 30]))
            
    random.shuffle(sizes)
    
    success_count = 0
    for idx, N in enumerate(sizes):
        case_id = start_id + idx
        out_dir = os.path.join(base_dir, str(case_id))
        ok = generate_single_test_case(out_dir, N, case_id, seed=10000 + case_id)
        if ok:
            success_count += 1
            if (idx + 1) % 100 == 0 or idx + 1 == total_count:
                print(f"Progress: {idx + 1}/{total_count} test cases created and validated.")
        else:
            print(f"Failed to generate valid test case {case_id} after retries.")
            
    print(f"\nCompleted! Successfully generated and verified {success_count}/{total_count} test cases.")

if __name__ == '__main__':
    generate_1000_tests()
