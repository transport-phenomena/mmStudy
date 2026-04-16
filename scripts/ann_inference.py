#!/usr/bin/env python3
"""
ANN inference helper for C++ integration.
Loads the trained model and applies it to compute dR correction.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Load model from parent directory (relative to this script)
script_dir = Path(__file__).parent
model_path = script_dir.parent.parent / "modelParameters.pt"

class MatrixNet(nn.Module):
    def __init__(self, maxN):
        super().__init__()
        out = 3 * 3 * maxN
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, out)
        )
        self.maxN = maxN

    def forward(self, x):
        y = self.net(x)
        return y.view(-1, 3, 3 * self.maxN)

def load_model():
    """Load the trained ANN model."""
    model = MatrixNet(50)
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def run_inference(phi, n_particles, maxN=50):
    """
    Run inference to get ANN correction dR.
    
    Args:
        phi: volume fraction
        n_particles: number of particles
        maxN: maximum number of particles in model
        
    Returns:
        List[List[float]]: 3x(3*maxN) matrix as nested list
    """
    model = load_model()
    
    log_phi = np.log10(phi)
    n_norm = n_particles / maxN
    input_tensor = torch.tensor([[log_phi, n_norm]], dtype=torch.float32)
    
    with torch.no_grad():
        dR = model(input_tensor)
    
    # Convert to numpy and return as nested list
    dR_np = dR.squeeze(0).cpu().numpy()
    return dR_np.tolist()

if __name__ == "__main__":
    # Read input from stdin as JSON
    try:
        input_data = json.loads(sys.stdin.read())
        phi = input_data['phi']
        n_particles = input_data['nParticles']
        
        dR = run_inference(phi, n_particles)
        
        # Output as JSON
        print(json.dumps(dR))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
