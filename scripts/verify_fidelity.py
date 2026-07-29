import json

import numpy as np

with open("results/real_machine/smoke_quick.json", encoding="utf-8") as f:
    smoke = json.load(f)

bits = smoke["h_gate"]["raw"]["resultStatus"]
flat_bits = np.array([b[0] for b in bits])
p1_real = np.mean(flat_bits)
p0_real = 1 - p1_real
noise_strength = abs(0.5 - p0_real) * 2
fidelity = 1 - noise_strength

print("=== 真机噪声数据验证 ===")
print(f"Task ID: {smoke['h_gate']['task_id']}")
print(f"Shots: {len(flat_bits)}")
print(f"P(0) = {p0_real:.6f}")
print(f"P(1) = {p1_real:.6f}")
print("理想值 = 0.5")
print(f"噪声强度 = {noise_strength:.6f}")
print(f"保真度 = {fidelity:.6f}")
print()
print("=== 验证结论 ===")
print(f"真实保真度 = {fidelity:.4f} (约{round(fidelity * 100, 1)}%)")
print(f"Gemini声称的0.976 与实际计算的{fidelity:.4f} 不符")
print(f"差异 = {abs(0.976 - fidelity):.4f} (约{round(abs(0.976 - fidelity) * 100, 1)}个百分点)")
