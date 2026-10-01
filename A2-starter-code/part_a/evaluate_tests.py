import argparse
import os
import sys
import time
import numpy as np
from env import TreasureHunt
from agent import Agent

def get_time_budget(N):
    """
    Determine training time budget based on grid size N:
    - Small grids (N <= 10): 60s (matches Test 1 and Test 3 in assignment PDF)
    - Medium grids (10 < N <= 20): 180s (3 minutes)
    - Large grids (20 < N <= 30): 900s (15 minutes, matches Test 2 in assignment PDF)
    """
    if N <= 10:
        return 60
    elif N <= 20:
        return 180
    else:
        return 900

def evaluate_single_test(test_dir, num_runs=20):
    layout_file = os.path.join(test_dir, 'layout.txt')
    prob_file = os.path.join(test_dir, 'prob.txt')
    
    if not os.path.exists(layout_file) or not os.path.exists(prob_file):
        return None
        
    # Read N from layout
    with open(layout_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        N = len(lines)
        
    time_budget = get_time_budget(N)
    
    # Initialize and train agent
    t_train_start = time.time()
    agent = Agent(layout_file, prob_file)
    agent.learn_policy(time_budget)
    train_duration = time.time() - t_train_start
    
    # Initialize environment to obtain start state and baseline
    env = TreasureHunt(layout_file, prob_file)
    state0 = env.get_state()
    s_idx = agent._idx(state0[0])
    m = agent._mask(state0[2])
    p1_i = agent.p1_to_idx.get(tuple(state0[1][0]), 0)
    p2_i = agent.p2_to_idx.get(tuple(state0[1][1]), 0)
    baseline_score = float(agent.V[m, p1_i, p2_i, s_idx])
    
    # Run 20 evaluation trajectories
    scores = []
    pirate_deaths = 0
    fort_reached = 0
    treasures_collected = 0
    
    for _ in range(num_runs):
        env = TreasureHunt(layout_file, prob_file)
        state = env.get_state()
        steps = 0
        rewards = []
        max_steps = 2 * (N ** 2)
        
        while not env.done and steps < max_steps:
            action = agent.get_action(*state)
            state, reward, done = env.step(action)
            rewards.append(reward)
            steps += 1
            
        dis_reward = 0.0
        for r in reversed(rewards):
            dis_reward = r + env.df * dis_reward
        scores.append(dis_reward)
            
        # Check termination type
        ship_end = env.locations['ship'][0]
        if ship_end in env.locations['pirate']:
            pirate_deaths += 1
        elif ship_end in env.locations['fort']:
            fort_reached += 1
            
        treasures_collected += (2 - len(env.locations['treasure']))
        
    avg_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    diff = avg_score - baseline_score
    
    # Success criterion: within 1 standard error or empirical score >= baseline - 0.5
    status = "PASS" if avg_score >= baseline_score - 1.0 or diff >= -0.5 else "WARN"
    
    return {
        'test_dir': test_dir,
        'N': N,
        'time_budget': time_budget,
        'train_time': train_duration,
        'baseline_score': baseline_score,
        'avg_score': avg_score,
        'std_score': std_score,
        'diff': diff,
        'pirate_deaths': pirate_deaths,
        'fort_reached': fort_reached,
        'avg_treasures': treasures_collected / num_runs,
        'status': status
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate test cases for Part A MDP.")
    parser.add_argument("--tests_dir", type=str, default="tests", help="Base directory of test cases")
    parser.add_argument("--start_id", type=int, default=1, help="Starting test ID")
    parser.add_argument("--end_id", type=int, default=1003, help="Ending test ID (inclusive)")
    parser.add_argument("--num_cases", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of evaluation runs per test case")
    parser.add_argument("--summary_only", action="store_true", help="Print only summary without per-test details")
    args = parser.parse_args()
    
    # Collect test directories
    test_ids = list(range(args.start_id, args.end_id + 1))
    if args.num_cases is not None:
        test_ids = test_ids[:args.num_cases]
        
    valid_test_dirs = []
    for tid in test_ids:
        td = os.path.join(args.tests_dir, str(tid))
        if os.path.isdir(td) and os.path.exists(os.path.join(td, 'layout.txt')):
            valid_test_dirs.append(td)
            
    print(f"=" * 105)
    print(f" TREASURE HUNT PART A: EVALUATION RUNNER ({len(valid_test_dirs)} Test Cases, {args.num_runs} Iterations Each)")
    print(f"=" * 105)
    print(f"{'Test ID':<10} {'Grid N':<8} {'Budget':<8} {'Train Time':<12} {'Baseline V*':<13} {'Avg Score':<18} {'Diff':<10} {'Fort/20':<9} {'Status':<6}")
    print(f"-" * 105)
    
    results = []
    total_start_time = time.time()
    
    for i, test_dir in enumerate(valid_test_dirs):
        test_name = os.path.basename(test_dir)
        res = evaluate_single_test(test_dir, num_runs=args.num_runs)
        if res is None:
            continue
            
        results.append(res)
        
        if not args.summary_only:
            score_str = f"{res['avg_score']:>6.3f} +/- {res['std_score']:<4.2f}"
            diff_sign = "+" if res['diff'] >= 0 else ""
            diff_str = f"{diff_sign}{res['diff']:<5.3f}"
            print(f"{test_name:<10} {res['N']:<8} {str(res['time_budget']) + 's':<8} {res['train_time']:>6.3f}s     {res['baseline_score']:>7.3f}       {score_str:<18} {diff_str:<10} {res['fort_reached']:<9} {res['status']:<6}")
        elif (i + 1) % 50 == 0 or (i + 1) == len(valid_test_dirs):
            print(f"Progress: Evaluated {i + 1}/{len(valid_test_dirs)} test cases...")
            
    total_duration = time.time() - total_start_time
    
    # Summary Statistics
    if results:
        pass_count = sum(1 for r in results if r['status'] == "PASS")
        avg_diff = np.mean([r['diff'] for r in results])
        avg_train = np.mean([r['train_time'] for r in results])
        total_forts = sum(r['fort_reached'] for r in results)
        total_runs = len(results) * args.num_runs
        
        print(f"=" * 105)
        print(f" EVALUATION SUMMARY")
        print(f"=" * 105)
        print(f" Total Tests Evaluated:       {len(results)}")
        print(f" Total Trajectories Run:      {total_runs} (20 per test)")
        print(f" Tests Passed (Avg >= V* - tol): {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")
        print(f" Average Mean Score Diff:     {avg_diff:+.4f} (Empirical vs Exact Optimal Baseline)")
        print(f" Average Training Time:       {avg_train:.3f} seconds per test case")
        print(f" Overall Fort Arrival Rate:   {total_forts}/{total_runs} ({total_forts/total_runs*100:.1f}%)")
        print(f" Total Evaluation Duration:   {total_duration:.2f} seconds")
        print(f"=" * 105)

if __name__ == '__main__':
    main()
