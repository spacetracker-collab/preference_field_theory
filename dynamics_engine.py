import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pft_model import PreferenceFieldNet

def get_hamiltonian_flow(model, x, p):
    if not x.requires_grad: x.requires_grad_(True)
    if not p.requires_grad: p.requires_grad_(True)
    
    pref_intensity = model(x)
    H = 0.5 * torch.sum(p**2) - pref_intensity.sum()
    
    dx_dt = torch.autograd.grad(H, p, create_graph=True)[0]
    dp_dt = -torch.autograd.grad(H, x, create_graph=True)[0]
    
    return dx_dt, dp_dt

def simulate_alignment_lab(steps=60, dt=0.1, damping=0.05):
    model = PreferenceFieldNet(input_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    x = torch.tensor([[1.0, 1.0]], requires_grad=True) 
    p = torch.tensor([[0.2, -0.1]], requires_grad=True) 
    
    history_ai = []
    history_goal = []

    for t in range(steps):
        # Target drift
        target = torch.tensor([[2.0*torch.cos(torch.tensor(t*0.1)), 2.0*torch.sin(torch.tensor(t*0.1))]])
        history_goal.append(target.numpy())

        # Field update
        optimizer.zero_grad()
        loss = -model(target).sum()
        loss.backward()
        optimizer.step()

        # Hamiltonian Dynamics with Damping (to stop the rotation)
        dx, dp = get_hamiltonian_flow(model, x, p)
        with torch.no_grad():
            x += dx * dt
            # Damping prevents the infinite 'rotating' loop in phase space
            p = p * (1 - damping) + (dp * dt) 
            
        history_ai.append(x.clone().detach().numpy())

    return np.array(history_ai).squeeze(), np.array(history_goal).squeeze()
