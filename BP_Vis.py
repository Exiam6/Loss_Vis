import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 3)
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

torch.manual_seed(0)
X = torch.randn(100, 2)
y = (X[:, 0]**2 + X[:, 1]**2).unsqueeze(1)

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
param_history = []

def get_params():
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(params):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(params[idx:idx + numel].view(p.size()))
        idx += numel

def compute_loss_surface(x_range, y_range):
    w1 = np.linspace(x_range[0], x_range[1], 100)
    w2 = np.linspace(y_range[0], y_range[1], 100)
    loss_surface = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            params = get_params().clone()
            params[0] = w1[i]
            params[1] = w2[j]
            set_params(params)
            with torch.no_grad():
                output = model(X)
                loss = criterion(output, y)
                loss_surface[i, j] = loss.item()
    
    return w1, w2, loss_surface


save_dir = '/Users/zhaozifan/Desktop/ML_Playground/plots_2D' # Modify this to your file path
os.makedirs(save_dir, exist_ok=True)

# main
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    param_history.append(get_params().numpy().copy())
    params_start = param_history[0]
    params_end = param_history[-1]
    x_range = [min(params_start[0], params_end[0]) - 1, max(params_start[0], params_end[0]) + 1]
    y_range = [min(params_start[1], params_end[1]) - 1, max(params_start[1], params_end[1]) + 1]

    w1, w2, loss_surface = compute_loss_surface(x_range, y_range)

    # Plot
    plt.contourf(w1, w2, loss_surface, levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot([p[0] for p in param_history], [p[1] for p in param_history], 'r-o')
    plt.title(f'Loss Landscape and Gradient Descent Steps (Epoch {epoch + 1})')
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')

    plot_path = os.path.join(save_dir, f'epoch_{epoch + 1}.png')
    plt.savefig(plot_path)
    plt.close()

print(f'Plots saved in {save_dir}')
