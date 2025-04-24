# scaled_pattern_predictor.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# — USER‐TWEAKABLE BASE PARAMETERS —
base_params = {
    'a': 0.24,        # plate width (m)
    'b': 0.24,        # plate height (m)
    'h': 0.0008,      # thickness (m)
    'rho': 3125,      # density (kg/m³)
    'E': 1e9,         # Young’s modulus (Pa)
    'nu': 0.3,        # Poisson’s ratio
    'F0': 1.0,        # forcing amplitude (N)
    'c_damp': 5.0,    # damping coefficient
    'x0': 0.12,       # drive x (m)
    'y0': 0.12,       # drive y (m)
    'm_max': 10,      # modes in x
    'n_max': 10,      # modes in y
    'Nx': 200,        # grid points x
    'Ny': 200,        # grid points y
}

# Scales and frequency sweep
scales     = [1.0, 0.8, 0.6, 0.4, 0.2]
freq_range = (0.0, 300.0, 1.0)  # (f_min, f_max, f_step)

# Path to your target image
TARGET_PATH = "data_7.jpg"


def simulate_pattern(freq, params):
    """
    Compute the plate response Z(x,y) at drive frequency `freq` (Hz),
    using Kirchhoff–Love modal summation with given params.
    Returns the 2D absolute‐normalized pattern (Nx×Ny array).
    """
    # Unpack
    a, b = params['a'], params['b']
    h, rho, E, nu = params['h'], params['rho'], params['E'], params['nu']
    F0, c_damp = params['F0'], params['c_damp']
    x0, y0 = params['x0'], params['y0']
    m_max, n_max = params['m_max'], params['n_max']
    Nx, Ny = params['Nx'], params['Ny']

    # Derived constants
    rho_h = rho * h
    D     = E * h**3 / (12*(1 - nu**2))
    scale = np.sqrt(D / rho_h)

    # Modal eigenfrequencies and drive coefficients
    omegas = []
    phi0   = []
    for m in range(1, m_max+1):
        for n in range(1, n_max+1):
            k2 = (m/a)**2 + (n/b)**2
            ω  = (np.pi**2 * k2) * scale
            omegas.append(ω)
            phi0.append(np.sin(m*np.pi*x0/a) * np.sin(n*np.pi*y0/b))
    omegas = np.array(omegas)
    phi0   = np.array(phi0)

    # Spatial modes
    x = np.linspace(0, a, Nx)
    y = np.linspace(0, b, Ny)
    xx, yy = np.meshgrid(x, y)
    phis = np.stack([
        np.sin(m*np.pi*xx/a) * np.sin(n*np.pi*yy/b)
        for m in range(1, m_max+1)
        for n in range(1, n_max+1)
    ], axis=0)

    # Drive response
    omega = 2*np.pi * freq
    denom = rho_h * (omegas**2 - omega**2) + c_damp * omega
    A     = F0 * phi0 / denom      # modal amplitudes
    Z     = np.tensordot(A, phis, axes=(0,0))

    # Normalize absolute pattern
    Zabs  = np.abs(Z)
    Znorm = Zabs / Zabs.max()

    # Fade to white at edges
    dist_edge = np.minimum.reduce([xx, a-xx, yy, b-yy])
    w = np.clip(1 - (dist_edge * 2 / min(a, b)), 0, 1)
    I = Znorm*(1 - w) + w
    return np.clip(I, 0, 1)


def load_and_normalize(path, nx, ny):
    """
    Load grayscale image, resize to (nx,ny), normalize to [0,1].
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load '{path}'")
    resized = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
    f = resized.astype(np.float32)
    return (f - f.min()) / (f.max() - f.min())


def find_best_match():
    best = {'ssim': -1, 'frequency': None, 'scale': None}

    f_min, f_max, f_step = freq_range
    freqs = np.arange(f_min, f_max + f_step, f_step)

    for scale in scales:
        # Scale geometric and drive location
        params = base_params.copy()
        params.update({
            'a':    base_params['a'] * scale,
            'b':    base_params['b'] * scale,
            'x0':   base_params['x0'] * scale,
            'y0':   base_params['y0'] * scale,
        })

        # Preload target at this grid size
        target = load_and_normalize(TARGET_PATH, params['Nx'], params['Ny'])

        print(f"Scanning scale={scale:.1f} over {len(freqs)} freqs…")
        for f in freqs:
            pred = simulate_pattern(f, params)
            score = ssim(target, pred, data_range=1.0)
            if score > best['ssim']:
                best.update(ssim=score, frequency=f, scale=scale)

        print(f" → best@{scale:.1f} → {best['frequency']:.1f} Hz  (SSIM={best['ssim']:.4f})")

    return best


if __name__ == "__main__":
    result = find_best_match()
    print("\n🏆 Overall best match:")
    print(f"   Frequency:    {result['frequency']:.1f} Hz")
    print(f"   Scale factor: {result['scale']:.1f}")
    print(f"   SSIM score:   {result['ssim']:.4f}")
