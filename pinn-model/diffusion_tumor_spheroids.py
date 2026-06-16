# Sri Rama Jayam
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm # Import cm specifically
import torch.optim as optim
import matplotlib.animation as animation # Import for animation
from matplotlib.animation import FuncAnimation # Import for animation

# --- NEW: Import for data generation ---
from scipy.integrate import solve_ivp

# %matplotlib inline
# ^ Uncomment this if running in a Jupyter-like environment
# to display plots inline.

# ===================================================================
# --- Data Generation (from Previous Code) ---
# We run the 1D SciPy solver to get our "experimental" data
# ===================================================================

def dCdt_scipy(t, C, N, dr, D, K_sink):
    """
    This function is called by the ODE solver (SciPy).
    It returns the time derivative of C at each spatial node.
    """
    dC_dt = np.zeros(N)
    r_nodes = np.linspace(0, 1.0, N)

    # Loop over all INTERIOR nodes (from i=0 to i=N-2)
    for i in range(N - 1):
        if i == 0:
            # Handle the center node (i=0) using L'Hopital's rule
            dC_dt[i] = 4 * D * (C[1] - C[0]) / dr**2 - K_sink
        else:
            # Handle all other interior nodes (i > 0)
            C_left, C_mid, C_right = C[i-1], C[i], C[i+1]
            r = r_nodes[i]
            d2C_dr2 = (C_right - 2*C_mid + C_left) / dr**2
            dC_dr = (C_right - C_left) / (2 * dr)
            dC_dt[i] = D * (d2C_dr2 + (1/r) * dC_dr) - K_sink

    # Handle the boundary node (i=N-1) - Since we take Dirichlet boundary condition
    dC_dt[N-1] = 0.0
    return dC_dt

def generate_experimental_data_and_points(t_range=(0, 2), radius=1.0, K_sink_data = 0.01):
    """
    Generates the "experimental" data using the SciPy solver, adds noise,
    and converts the (r, t) points to (x, y, t) training points for the PINN.
    """
    print("Generating noisy 'experimental' data using SciPy...")

    # --- 1. Define Parameters for data generation ---
    D_data = 0.1 # Diffusion coefficient (must match PINN's alpha)
    a_data = radius # Radius of the circle
    N_data = 50 # Number of spatial points (nodes)
    noise_level = 0.05 # Standard deviation of the noise
    t_span_data = t_range
    t_points_data = np.linspace(t_span_data[0], t_span_data[1], num=100) # 100 time points

    # --- 2. Create the spatial grid for data ---
    r_nodes_data = np.linspace(0, a_data, N_data)
    dr_data = r_nodes_data[1] - r_nodes_data[0]

    # --- 3. Set Initial and Boundary Conditions for data ---
    C_initial_data = np.zeros(N_data)
    C_initial_data[N_data-1] = 1.0

    # --- 4. Solve the ODE system ---
    sol = solve_ivp(
        lambda t, C: dCdt_scipy(t, C, N_data, dr_data, D_data, K_sink_data),
        t_span_data,
        C_initial_data,
        t_eval=t_points_data,
        method='BDF'
    )

    C_clean_matrix = sol.y
    t_vec = sol.t

    # --- 5. Add Noise ---
    noise = np.random.normal(loc=0.0, scale=noise_level, size=C_clean_matrix.shape)
    C_experimental_matrix = C_clean_matrix + noise
    C_experimental_matrix = np.maximum(0, C_experimental_matrix) # No negative conc.

    # --- 6. Prepare data for PINN ---
    # We will use all data points EXCEPT t=0, as t=0 is handled
    # by the `initial_condition_loss`.

    r_data_pinn = torch.tensor(r_nodes_data, dtype=torch.float32)
    t_data_pinn = torch.tensor(t_vec[1:], dtype=torch.float32) # Exclude t=0
    C_data_pinn = torch.tensor(C_experimental_matrix[:, 1:], dtype=torch.float32) # Exclude t=0

    # Create a meshgrid of (r, t) points
    R, T = torch.meshgrid(r_data_pinn, t_data_pinn, indexing='ij')

    # Flatten them
    R_flat = R.reshape(-1, 1)
    T_flat = T.reshape(-1, 1)
    C_target_flat = C_data_pinn.reshape(-1, 1)

    # Convert (r, t) points to (x, y, t) points
    # We use random angles to enforce radial symmetry
    num_data_points = R_flat.shape[0]
    Theta_rand = torch.rand(num_data_points, 1) * 2 * np.pi

    X_data = R_flat * torch.cos(Theta_rand)
    Y_data = R_flat * torch.sin(Theta_rand)

    print(f"Generated {num_data_points} (x, y, t) data points for training.")

    return X_data, Y_data, T_flat, C_target_flat
# ===================================================================
# --- PINN Definition ---
# ===================================================================

class Diffusion2DPINN(nn.Module):
    def __init__(self, num_layers=4, n_hidden=40):
        super(Diffusion2DPINN, self).__init__()

        layers = []
        layers.append(nn.Linear(3, n_hidden)) # Input: (x, y, t)
        layers.append(nn.Tanh())
        for _ in range(num_layers-1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(n_hidden, 1)) # Output: c(x, y, t)
        self.net = nn.Sequential(*layers)

        self.alpha = 0.1 # Diffusion coefficient
        self.K_sink = 0.01 # Sink term

    def forward(self, x, y, t):
        input = torch.cat((x, y, t), dim=1)
        return self.net(input)


    def compute_derivatives(self, x, y, t):
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u = self.forward(x, y, t)

        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
        d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]

        return du_dt, d2u_dx2, d2u_dy2

    def pde_loss(self, x, y, t):
        du_dt, d2u_dx2, d2u_dy2 = self.compute_derivatives(x, y, t)
        # Residual of the PDE: du/dt - alpha * (d2u/dx2 + d2u/dy2)
        return torch.mean((du_dt - self.alpha * (d2u_dx2 + d2u_dy2) + self.K_sink)**2)

    def initial_condition_loss(self, x_initial, y_initial, t_initial, u_initial):
        u_pred = self.forward(x_initial, y_initial, t_initial)
        return torch.mean((u_pred - u_initial)**2)

    def boundary_condition_loss(self, x_boundary, y_boundary, t_boundary, u_boundary):
        u_pred = self.forward(x_boundary, y_boundary, t_boundary)
        return torch.mean((u_pred - u_boundary)**2)

    # --- NEW LOSS FUNCTION ---
    def data_loss(self, x_data, y_data, t_data, u_data_target):
        """
        Computes the loss against the "experimental" data points.
        """
        u_pred = self.forward(x_data, y_data, t_data)
        return torch.mean((u_pred - u_data_target)**2)


# ===================================================================
# --- Training Point Generation (for PDE, IC, BC) ---
# ===================================================================

def generate_circle_domain(n_r=15, n_theta=30, n_t=10, radius=1.0, t_range=(0, 2)):
    """Generate training points in a circular domain."""
    r = torch.linspace(0, radius, n_r)
    theta = torch.linspace(0, 2*np.pi, n_theta)
    t = torch.linspace(t_range[0], t_range[1], n_t)
    R, THETA, T = torch.meshgrid(r, theta, t, indexing='ij')
    X = R * torch.cos(THETA)
    Y = R * torch.sin(THETA)
    return X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)

def generate_circle_boundary(n_theta=50, n_t=10, radius=1.0, t_range=(0, 2)):
    """Generate points on the circular boundary."""
    theta = torch.linspace(0, 2*np.pi, n_theta)
    t = torch.linspace(t_range[0], t_range[1], n_t)
    THETA, T = torch.meshgrid(theta, t, indexing='ij')
    X = radius * torch.cos(THETA)
    Y = radius * torch.sin(THETA)
    return X.reshape(-1, 1), Y.reshape(-1, 1), T.reshape(-1, 1)

def initial_condition(x, y, x0=0.0, y0=0.0, c_boundary=1.0, radius=1.0):
    """
    Defines the initial condition (0 inside, 1 on boundary).
    """
    r = torch.sqrt((x - x0)**2 + (y - y0)**2)
    # Use a small tolerance for the boundary
    return torch.where(r < radius - 1e-5, torch.zeros_like(x), torch.full_like(x, c_boundary))

# ===================================================================
# --- MODIFIED Training Function ---
# ===================================================================

def train_pinn(model, data_tensors, num_epochs=2000, learning_rate=0.001, data_loss_weight=10.0):
    """Train the PINN model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # --- Get Physics/Boundary training points ---
    x_train, y_train, t_train = generate_circle_domain()
    x_boundary, y_boundary, t_boundary = generate_circle_boundary()
    u_boundary = torch.full_like(x_boundary, 1.0)

    radius = 1.0
    n_points = 100
    x = torch.linspace(-radius, radius, n_points)
    y = torch.linspace(-radius, radius, n_points)
    X_init, Y_init = torch.meshgrid(x, y, indexing='ij')
    mask = X_init**2 + Y_init**2 <= radius**2
    x_initial = X_init[mask].reshape(-1, 1)
    y_initial = Y_init[mask].reshape(-1, 1)
    t_initial = torch.zeros_like(x_initial)
    u_initial = initial_condition(x_initial, y_initial)

    # --- Get Experimental Data training points ---
    x_data, y_data, t_data, u_data_target = data_tensors

    loss_history = []
    print("Starting training...")
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Compute PDE loss (interior points)
        pde_loss = model.pde_loss(x_train, y_train, t_train)

        # Compute boundary condition loss
        bc_loss = model.boundary_condition_loss(x_boundary, y_boundary, t_boundary, u_boundary)

        # Compute initial condition loss
        ic_loss = model.initial_condition_loss(x_initial, y_initial, t_initial, u_initial)

        # --- NEW: Compute data loss ---
        data_loss = model.data_loss(x_data, y_data, t_data, u_data_target)

        # Total loss (weighted)
        # We weight the BC, IC, and Data losses more heavily
        total_loss = pde_loss + 10.0 * bc_loss + 10.0 * ic_loss + data_loss_weight * data_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(total_loss.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item():.6f}, PDE: {pde_loss.item():.6f}, BC: {bc_loss.item():.6f}, IC: {ic_loss.item():.6f}, Data: {data_loss.item():.6f}')

    return loss_history

# ===================================================================
# --- Plotting and Animation Functions (Unchanged) ---
# ===================================================================

def plot_circular_solution(model, t_value=0.5, radius=1.0):
    """Plot the solution in a circular domain at a specific time."""
    n = 100
    x = torch.linspace(-radius, radius, n)
    y = torch.linspace(-radius, radius, n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    mask = X**2 + Y**2 <= radius**2
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_flat = torch.ones_like(x_flat) * t_value

    with torch.no_grad():
        u_pred = model(x_flat, y_flat, t_flat)
        U = u_pred.reshape(n, n)

    U_final = torch.full_like(U, 1.0)
    U_final[mask] = U[mask]

    vmin_val = 0.0
    vmax_val = 1.0


    plt.figure(figsize=(10, 8))
    plt.contourf(X.numpy(), Y.numpy(), U_final.numpy(), cmap=cm.viridis, levels=50, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar()
    cbar.set_label(r'Concentration (mmolm$^{-3}$)', fontsize=14)

    plt.xlabel('x (100 µm units)')
    plt.xlabel('y (100 µm units)')
    plt.title(f'Solution at t = {t_value}s')
    plt.axis('equal')
    circle = plt.Circle((0, 0), radius, fill=False, color='k')
    plt.gca().add_patch(circle)
    plt.tight_layout()
    plt.show()

def animate_solution(model, radius=1.0, t_range=(0, 2), num_frames=50):
    """Animates the solution over time and saves it as a GIF."""
    print("Generating animation... This may take a moment. ")

    fig, ax = plt.subplots(figsize=(10, 8))

    n = 100
    x = torch.linspace(-radius, radius, n)
    y = torch.linspace(-radius, radius, n)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    mask = X**2 + Y**2 <= radius**2
    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)
    t_values = torch.linspace(t_range[0], t_range[1], num_frames)

    dummy_levels = np.linspace(0, 1, 50)
    contour = ax.contourf(X.numpy(), Y.numpy(), X.numpy(), levels=dummy_levels, cmap=cm.viridis, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(contour, ax=ax, label='Concentration')

    def update(frame):
        ax.clear()
        t_val = t_values[frame].item()
        t_flat = torch.ones_like(x_flat) * t_val

        with torch.no_grad():
            u_pred = model(x_flat, y_flat, t_flat)
            U = u_pred.reshape(n, n)

        U_final = torch.full_like(U,1.0)
        U_final[mask] = U[mask]

        ax.contourf(X.numpy(), Y.numpy(), U_final.numpy(), cmap=cm.viridis, levels=50, vmin=0.0, vmax=1.0)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Solution at t = {t_val:.2f}s')
        ax.axis('equal')
        circle = plt.Circle((0, 0), radius, fill=False, color='k')
        ax.add_patch(circle)

    ani = FuncAnimation(fig, update, frames=len(t_values), blit=False)

    try:
        ani.save('diffusion_animation_pinn_with_data.gif', writer='pillow', fps=10)
        print("Animation saved as 'diffusion_animation_pinn_with_data.gif' ")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("You may need to install 'pillow': pip install pillow")

    plt.close(fig)

# ===================================================================
# --- MODIFIED Main Execution ---
# ===================================================================

def main():
    # --- 1. Generate the "Experimental" Data ---
    # This data will be used to guide the PINN training
    K_sink = 0.01
    t_range_global = (0, 2)
    radius_global = 1.0
    data_tensors = generate_experimental_data_and_points(t_range=t_range_global, radius=radius_global, K_sink_data = K_sink)

    # --- 2. Create and train the model ---
    model = Diffusion2DPINN()
    # Pass the experimental data to the training function
    loss_history = train_pinn(
        model,
        data_tensors,
        num_epochs=3000,
        data_loss_weight=20.0 # Give the data loss a significant weight
    )

    # --- 3. Plot results ---
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.show()

    # Plot solutions at different time points
    for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.00]:
        plot_circular_solution(model, t_value=t, radius=radius_global)

    # --- 4. Call the animation function ---
    animate_solution(model, radius=radius_global, t_range=t_range_global, num_frames=50)

if __name__ == "__main__":
    main()
