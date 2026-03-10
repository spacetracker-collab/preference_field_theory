import torch
import torch.nn as nn

class PreferenceFieldNet(nn.Module):
    def __init__(self, input_dim=2):
        super(PreferenceFieldNet, self).__init__()
        # Represents the Scalar Preference Field P(x)
        # Tanh activations ensure the field is smooth and twice-differentiable 
        # as required by Hamiltonian Dynamics (Eq 17).
        self.field = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # Scalar Preference Intensity
        )

    def forward(self, x):
        return self.field(x)

    def get_gradient(self, x):
        """
        Implementation of the Gradient Decision Principle (Section 3).
        Calculates ∇P(x), the direction of steepest preference increase.
        """
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        
        p = self.forward(x)
        # Calculate the gradient of the scalar field with respect to the input
        grad = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
        return grad
