#!/usr/bin/env python
"""一键演示量子电路模板生成器"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum.circuit_templates import (
    generate_bell_state,
    generate_ghz_state,
    generate_qaoa_circuit,
    generate_vqe_circuit,
)


def main():
    print("=" * 60)
    print("Bell 态电路 (Q0, Q1):")
    print(generate_bell_state())
    print("\n" + "=" * 60)
    print("GHZ-3 态电路:")
    print(generate_ghz_state(3))
    print("\n" + "=" * 60)
    print("VQE-4 电路 (depth=1):")
    print(generate_vqe_circuit(n_qubits=4, depth=1))
    print("\n" + "=" * 60)
    print("QAOA-5 电路 (p=1):")
    print(generate_qaoa_circuit(n_qubits=5, p_layers=1))


if __name__ == "__main__":
    main()
