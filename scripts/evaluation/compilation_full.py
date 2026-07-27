"""
量子电路编译AI优化 — 端到端：baseline + 训练 + 评估 + OR-Tools对比
"""
import sys, json, time, os, numpy as np
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

from qiskit.circuit.random import random_circuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ortools.sat.python import cp_model
from src.quantum.compilation_env import QuantumCompilationEnv
from src.scheduler.env import QuantumSchedulingEnv
from scripts.evaluation.run_simulation import SimulationEnv, SimulationTaskGenerator, FCFSStrategy, ShortestJobFirstStrategy, PPOStrategy, run_strategy

COUPLING = CouplingMap([(i, i + 1) for i in range(15)] + [(i + 1, i) for i in range(15)])

# ── 1. SABRE Baseline ──
print("=" * 60)
print("  [1/4] SABRE Baseline")
print("=" * 60)
sabre_swaps = []
for cat, cfg in {"sh": (5,8,5,10), "md": (9,14,10,20), "dp": (14,16,20,30)}.items():
    for i in range(20):
        qc = random_circuit(np.random.randint(*cfg[:2]), np.random.randint(*cfg[2:]), measure=False)
        pm = PassManager([SabreLayout(COUPLING, swap_trials=8, layout_trials=8), SabreSwap(COUPLING, trials=8)])
        compiled = pm.run(qc)
        sabre_swaps.append(compiled.count_ops().get("swap", 0))
sabre_avg = np.mean(sabre_swaps)
print(f"SABRE avg SWAP: {sabre_avg:.1f} (60 circuits)")

# ── 2. Train PPO ──
print("\n" + "=" * 60)
print("  [2/4] Train PPO Compilation Agent")
print("=" * 60)
class CircuitPool:
    def __init__(self, n=80): self.circuits = [random_circuit(np.random.randint(5,13), np.random.randint(5,21), measure=False) for _ in range(n)]; self.idx = 0
    def sample(self): self.idx += 1; return self.circuits[self.idx % len(self.circuits)]
pool = CircuitPool()
env = DummyVecEnv([lambda: QuantumCompilationEnv(pool.sample())])
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, verbose=0)
t0 = time.time()
model.learn(total_timesteps=50000)
print(f"Done: {time.time()-t0:.0f}s")
os.makedirs("deliverable_models", exist_ok=True)
model.save("deliverable_models/ppo_compilation_agent.zip")

# ── 3. Evaluate PPO ──
print("\n" + "=" * 60)
print("  [3/4] PPO vs SABRE")
print("=" * 60)
ppo_swaps = []
for i in range(20):
    qc = random_circuit(np.random.randint(5,13), np.random.randint(5,21), measure=False)
    env2 = QuantumCompilationEnv(qc)
    obs, _ = env2.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env2.step(int(action))
        if terminated or truncated: break
    ppo_swaps.append(env2.get_stats()["swap_count"])
ppo_avg = np.mean(ppo_swaps)
improvement = (1 - ppo_avg / max(1, sabre_avg)) * 100
print(f"PPO avg SWAP: {ppo_avg:.1f} vs SABRE: {sabre_avg:.1f} -> -{improvement:.1f}%")

# ── 4. OR-Tools comparison ──
print("\n" + "=" * 60)
print("  [4/4] OR-Tools vs RL Scheduling")
print("=" * 60)
def cp_sat(tasks, tl=60):
    m = cp_model.CpModel(); h = sum(max(1, t[1]) for t in tasks); s, e, iv = [], [], []
    for i, (_, d) in enumerate(tasks):
        si = m.NewIntVar(0, h, f"s{i}"); ei = m.NewIntVar(0, h, f"e{i}")
        s.append(si); e.append(ei); iv.append(m.NewIntervalVar(si, d, ei, f"iv{i}"))
        m.Add(ei <= m.NewIntVar(0, h, "mk"))
    m.AddNoOverlap(iv); mk = m.NewIntVar(0, h, "mk")
    for i in range(len(tasks)): m.Add(e[i] <= mk)
    m.Minimize(mk)
    sol = cp_model.CpSolver(); sol.parameters.max_time_in_seconds = tl; sol.Solve(m)
    return sol.ObjectiveValue()
ppo_load = PPO.load("deliverable_models/ppo_best_model_14dim.zip")
for n in [20, 50, 100]:
    tg = SimulationTaskGenerator(seed=42); tks = tg.generate_batch(n)[:n]
    ort_in = [(i, t.get("qubit_count", 0) * 10 + 5 if t.get("task_type") == "quantum" else 5) for i, t in enumerate(tks)]
    ort_r = cp_sat(ort_in)
    env3 = QuantumSchedulingEnv(max_steps=n, max_qubits=287, seed=42)
    se = SimulationEnv(env=env3, task_generator=SimulationTaskGenerator(seed=42))
    pf = run_strategy(se, PPOStrategy(ppo_load), num_episodes=3, tasks_per_episode=n, max_steps=n, verbose=False)
    ff = run_strategy(se, FCFSStrategy(), num_episodes=3, tasks_per_episode=n, max_steps=n, verbose=False)
    print(f"  {n} tasks: OR-Tools makespan={ort_r}, PPO reward={pf['avg_reward']:.0f}, FCFS={ff['avg_reward']:.0f}")

# ── Save ──
os.makedirs("results/compilation", exist_ok=True)
ts = time.strftime("%Y%m%d_%H%M%S")
result = {"ppo_avg_swap": ppo_avg, "sabre_avg_swap": sabre_avg, "improvement_pct": round(improvement, 1), "model": "ppo_compilation_agent.zip"}
with open(f"results/compilation/full_eval_{ts}.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n[OK] results/compilation/full_eval_{ts}.json")
