# Clear previous runs
!rm -rf preference_field_theory
!git clone https://github.com/spacetracker-collab/preference_field_theory.git
%cd preference_field_theory

import torch
import matplotlib.pyplot as plt
from dynamics_engine import simulate_alignment_lab

# Execute the PFT Lab
print("Calculating Phase-Space Trajectory...")
ai_path, goal_path = simulate_alignment_lab(steps=100, dt=0.1)

# Visualization
plt.figure(figsize=(10, 8))
plt.plot(goal_path[:, 0], goal_path[:, 1], 'r--', alpha=0.5, label='Human Preference Drift')
plt.plot(ai_path[:, 0], ai_path[:, 1], 'b-', linewidth=2, label='AI Hamiltonian Path (Eq 17)')
plt.scatter(ai_path[-1, 0], ai_path[-1, 1], color='blue', s=100, label='Final AI State')



plt.title("Convergence Analysis: Hamiltonian Alignment with Damping")
plt.xlabel("Cognitive State X1")
plt.ylabel("Cognitive State X2")
plt.legend()
plt.grid(True)
plt.show()
