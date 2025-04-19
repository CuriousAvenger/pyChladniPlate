import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyopencl as cl
import sys

# — Physical and modal parameters —
a, b = 0.24, 0.24       # Plate dimensions (m)
h, rho, E, nu = 0.002, 7850, 2.1e11, 0.3
D = E * h**3 / (12 * (1 - nu**2))
F0, c_damp = 1.0, 5.0   # Forcing amplitude & damping
x0, y0 = a/2, b/2       # Drive location

# — Animation settings — 
f_max = 850             # Hz cap
frames = 600            # more frames for super smooth
interval = 40           # ms pause → ~25 FPS

# — Spatial grid —
Nx, Ny = 200, 200
x = np.linspace(0, a, Nx).astype(np.float32)
y = np.linspace(0, b, Ny).astype(np.float32)
xx, yy = np.meshgrid(x, y)
xx_flat = xx.ravel()
yy_flat = yy.ravel()

# — Precompute natural frequencies & drive projections —
omegas = []
phi0   = []
for m in range(1, 31):
    for n in range(1, 31):
        ω = (np.pi**2) * np.sqrt(D/(rho*h)) * np.sqrt((m/a)**2 + (n/b)**2)
        omegas.append(ω)
        phi0.append(np.sin(m*np.pi*x0/a) * np.sin(n*np.pi*y0/b))
omegas    = np.array(omegas, dtype=np.float32)
phi0      = np.array(phi0,   dtype=np.float32)
num_modes = omegas.size

# — OpenCL setup on platform 0, device 0 —
plat  = cl.get_platforms()[0]
dev   = plat.get_devices()[0]
ctx   = cl.Context([dev])
queue = cl.CommandQueue(ctx)
mf    = cl.mem_flags

omegas_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omegas)
phi0_buf   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phi0)
xx_buf     = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=xx_flat)
yy_buf     = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=yy_flat)
Z_buf      = cl.Buffer(ctx, mf.WRITE_ONLY, xx_flat.nbytes)

# — OpenCL kernel (real‑only denominator) —
kernel_source = """
__kernel void compute_Z(
    const float f,
    const int num_modes,
    __global const float *omegas,
    __global const float *phi0,
    __global const float *xx,
    __global const float *yy,
    const float a,
    const float b,
    const float rho_h,
    const float c_damp,
    const float F0,
    __global float *Z)
{
    int gid = get_global_id(0);
    float x = xx[gid], y = yy[gid];
    float sum = 0.0f;
    float omega = 2.0f * 3.14159265f * f;

    for(int k = 0; k < num_modes; k++) {
        int m = (k / 30) + 1;
        int n = (k % 30) + 1;
        float phi = sin(m * 3.14159265f * x / a)
                  * sin(n * 3.14159265f * y / b);
        float delta = omegas[k]*omegas[k] - omega*omega;
        float denom = rho_h * delta + c_damp * omega;
        float A = F0 * phi0[k] / denom;
        sum += A * phi;
    }
    Z[gid] = sum;
}
"""
prg = cl.Program(ctx, kernel_source).build()

# — Host constants for kernel launch —
rho_h = np.float32(rho * h)
a32, b32 = np.float32(a), np.float32(b)

# — Matplotlib setup —
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

def update(frame):
    f = frame * f_max / (frames - 1)
    prg.compute_Z(
        queue,
        (Nx * Ny,),
        None,
        np.float32(f),
        np.int32(num_modes),
        omegas_buf,
        phi0_buf,
        xx_buf,
        yy_buf,
        a32, b32,
        rho_h,
        np.float32(c_damp),
        np.float32(F0),
        Z_buf
    )
    Z_flat = np.empty(Nx*Ny, dtype=np.float32)
    cl.enqueue_copy(queue, Z_flat, Z_buf)
    Z = Z_flat.reshape((Ny, Nx))

    ax.clear()
    ax.contour(x, y, Z, levels=[0], colors='k')
    ax.set_title(f"Chladni Nodal Lines at {f:.1f} Hz")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal')
    return ax.collections

ani = animation.FuncAnimation(
    fig, update,
    frames=frames,
    interval=interval,
    blit=False
)
plt.show()
