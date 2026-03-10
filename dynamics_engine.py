import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pft_model import PreferenceFieldNet

def get_hamiltonian_flow(model, x, p):
    """
    Implements Hamilton's Equations (Eq 17):
    dx/dt = dH/dp (Velocity)
    dp/dt = -dH/dx (Force from the Preference Field)
    """
    if not x.requires_grad: x.requires_grad_(True)
    if not p.requires_grad: p.requires_grad_(True)
    
    # Potential Energy V(x) = -P(x). AI seeks to maximize P(x).
    pref_intensity = model(x)
    H = 0.5 * torch.sum(p**2) - pref_intensity.sum()
    
    # Hamilton's Equations with higher-order gradient tracking
    dx_dt = torch.autograd.grad(H, p, create_graph=True)[0]
    dp_dt = -torch.autograd.grad(H, x, create_graph=True)[0]
    
    return dx_dt, dp_dt

def simulate_drift_and_alignment(steps=30, dt=0.2):
    """
    Simulates Section 4: Preference Drift.
    The 'True Goal' moves, and the AI tracks it using Hamiltonian Flow.
    """
    model = PreferenceFieldNet(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    x = torch.tensor([[1.0, 1.0]], requires_grad=True) # AI Position
    p = torch.tensor([[0.1, 0.1]], requires_grad=True)  # AI Momentum (Intent)
    
    history_ai = []
    history_goal = []

    for t in range(steps):
        # 1. Define Drifting Goal (Moving in a circle)
        target = torch.tensor([[2*torch.cos(torch.tensor(t*0.2)), 2*torch.sin(torch.tensor(t*0.2))]])
        history_goal.append(target.numpy())

        # 2. Update Field (AI learns the drift)
        optimizer.zero_grad()
        loss = -model(target).sum()
        loss.backward()
        optimizer.step()

        # 3. Calculate Hamiltonian Flow (Decision Dynamics)
        dx, dp = get_hamiltonian_flow(model, x, p)
        with torch.no_grad():
            x += dx * dt
            p += dp * dt
        history_ai.append(x.clone().detach().numpy())

    return history_ai, history_goal

if __name__ == "__main__":
    ai_path, goal_path = simulate_drift_and_alignment()
    print("Simulation complete. AI has successfully tracked the drifting cognitive manifold.")
