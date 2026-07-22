# Simulating Drug Transport Through Spheroids Using Physics-Informed Neural Networks

A Physics-Informed Neural Network (PINN) framework for solving the transient reaction-diffusion equation governing drug transport inside tumor spheroids. The project demonstrates how PINNs can model diffusion and metabolic drug consumption without requiring computational meshes, providing an efficient alternative to conventional CFD methods.

---

## Overview

Tumor spheroids are three-dimensional cellular structures widely used as in-vitro models for cancer research. Understanding drug diffusion through these spheroids is essential for optimizing drug delivery and therapeutic effectiveness.

Traditional numerical approaches such as finite difference or finite element methods require mesh generation and solve the governing equations only at discrete grid points. Physics-Informed Neural Networks overcome these limitations by embedding the governing PDE directly into the neural network's loss function, enabling continuous predictions while naturally incorporating physical constraints. :contentReference[oaicite:0]{index=0}

---

## Features

- Physics-Informed Neural Network implementation for reaction-diffusion systems
- Mesh-free solution of Fick's Second Law
- Supports sparse and noisy observational data
- Incorporates:
  - PDE residual loss
  - Boundary condition loss
  - Initial condition loss
  - Data loss
- Continuous prediction over space and time
- Visualization of concentration evolution during diffusion

---

## Mathematical Model

The transient reaction-diffusion equation governing drug concentration is

```text
∂C/∂t = D∇²C − K
```

where

- **C(x, y, t)** = Drug concentration
- **D** = Diffusion coefficient
- **K** = Constant metabolic sink term

Boundary condition:

```text
C(x, y, t) = C₀    on the spheroid boundary
```

Initial condition:

```text
C(x, y, 0) = 0     throughout the spheroid
```
---

## Neural Network Architecture

The PINN uses a fully-connected feedforward neural network:

- Input:
  - x-coordinate
  - y-coordinate
  - time
- Output:
  - Drug concentration

Architecture:

- 4 hidden layers
- 40 neurons per hidden layer
- Adam optimizer
- 3000 training epochs

The total loss is a weighted combination of:

- Data loss
- PDE residual loss
- Boundary condition loss
- Initial condition loss :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Methodology

1. Generate reference solutions using SciPy.
2. Corrupt the numerical solution with noise to simulate experimental measurements.
3. Train the PINN using:
   - Experimental data
   - Governing PDE
   - Initial conditions
   - Boundary conditions
4. Predict the continuous concentration field throughout the domain.

---

## Results

The trained PINN successfully learns the diffusion dynamics within the spheroid.

Key observations:

- Drug concentration remains highest near the boundary.
- Concentration gradually diffuses toward the center.
- The sink term models metabolic consumption, reducing concentration throughout the interior.
- Training converges successfully within 3000 epochs.
- The model accurately reproduces the expected reaction-diffusion behavior. :contentReference[oaicite:4]{index=4}

---

## Advantages of PINNs

Compared to traditional CFD approaches:

- No mesh generation required
- Continuous solution representation
- Naturally incorporates governing physics
- Effective with limited or noisy datasets
- Easily adaptable to inverse problems and parameter estimation

---

## Technologies Used

- Python
- PyTorch
- NumPy
- SciPy
- Matplotlib

---

## Applications

- Drug delivery simulation
- Tumor spheroid modeling
- Biomedical engineering
- Scientific machine learning
- Physics-informed deep learning
- Personalized medicine research

---

## Future Work

Potential extensions include:

- Three-dimensional spheroid simulations
- Variable diffusion coefficients
- Nonlinear reaction kinetics
- Multiple interacting drugs


---

## References

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A Deep Learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*

2. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations.*

---

## Acknowledgements

This work was carried out at the **Indian Institute of Technology Madras** and presented at the **19th Annual International Conference on Complex Fluids and Soft Matter (CompFlu 2025), Indian Institute of Science, Bengaluru.** 
