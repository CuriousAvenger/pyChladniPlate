import numpy as np
import matplotlib.pyplot as plt

# — USER‐TWEAKABLE PARAMETERS —
freq        = 228         # Drive frequency (Hz)
a, b        = 0.24, 0.24    # Plate dimensions (m)
h           = 0.0008        # Plate thickness (m)
rho, E, nu  = 3125, 1e9, 0.3
F0, c_damp  = 1.0, 5.0      # Drive amplitude (N), damping coeff
x0, y0      = a/2, b/2      # Drive location (m)
m_max,n_max = 10, 10        # Modes retained
Nx, Ny      = 200, 200      # Grid resolution

# — DERIVED CONSTANTS & MODAL PRECOMPUTATION —
rho_h = rho * h
D     = E * h**3/(12*(1-nu**2))
scale = np.sqrt(D/rho_h)

omegas = []
phi0   = []
for m in range(1, m_max+1):
    for n in range(1, n_max+1):
        k2 = (m/a)**2 + (n/b)**2
        ω  = (np.pi**2 * k2)*scale
        omegas.append(ω)
        phi0.append(np.sin(m*np.pi*x0/a)*np.sin(n*np.pi*y0/b))
omegas = np.array(omegas)
phi0   = np.array(phi0)

# — BUILD MODAL SHAPES —
x = np.linspace(0, a, Nx)
y = np.linspace(0, b, Ny)
xx, yy = np.meshgrid(x, y)
phis = np.stack([
    np.sin(m*np.pi*xx/a)*np.sin(n*np.pi*yy/b)
    for m in range(1, m_max+1)
    for n in range(1, n_max+1)
], axis=0)

# — COMPUTE Z(x,y) —
omega = 2*np.pi*freq
denom = rho_h*(omegas**2 - omega**2) + c_damp*omega
A     = F0 * phi0 / denom
Z     = np.tensordot(A, phis, axes=(0,0))

# — NORMALIZE & FADE OUT AT EDGES —
Zabs  = np.abs(Z)
Zmax  = Zabs.max()
Znorm = Zabs / Zmax

# distance to nearest edge:
dist_edge = np.minimum.reduce([xx, a-xx, yy, b-yy])
# normalized weight: 1 at edges, 0 at center
w = np.clip(1 - (dist_edge * 2 / np.min((a,b))), 0, 1)

# composite intensity: interior = Znorm, edges → white(1)
I = Znorm*(1 - w) + w
I = np.clip(I, 0, 1)

# — PLOT SMOOTH FADED HEATMAP + NODAL LINE —
plt.figure(figsize=(6,6))
im = plt.imshow(
    I,
    origin='lower',
    extent=(0, a, 0, b),
    cmap='Blues_r',
    aspect='equal',
    interpolation='bicubic'
)
plt.colorbar(im, label='Relative amplitude / fade weight')
# overlay nodal line Z=0 in contrasting color
# plt.contour(x, y, Z, levels=[0], colors='red', linewidths=2)

plt.title(f"Chladni Plate — {freq:.1f} Hz")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.show()
