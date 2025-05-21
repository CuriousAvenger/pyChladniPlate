#!/usr/bin/env python3
import numpy as np
from typing import Tuple

class ChladniPlate:
    """
    /**
     * Simulator for vibration and nodal‐line patterns of a simply‐supported rectangular plate
     * driven by a point force.
     */
    """

    def __init__(
        self,
        a: float,
        b: float,
        h: float,
        rho: float,
        E: float,
        nu: float,
        zeta: float
    ) -> None:
        """
        /**
         * Initialize plate geometry and material properties.
         *
         * @param a    Plate width (m).
         * @param b    Plate height (m).
         * @param h    Plate thickness (m).
         * @param rho  Material density (kg/m³).
         * @param E    Young's modulus (Pa).
         * @param nu   Poisson's ratio (dimensionless).
         * @param zeta Modal damping ratio (dimensionless).
         */
        """
        self.a: float = a
        self.b: float = b
        self.h: float = h
        self.rho: float = rho
        self.E: float = E
        self.nu: float = nu
        self.zeta: float = zeta

        # Flexural rigidity D = E·h³ / [12·(1 – ν²)]
        self.D: float = (self.E * self.h**3) / (12 * (1 - self.nu**2))

    def compute_contours(
        self,
        freq: float,
        F0: float,
        x0: float,
        y0: float,
        num_points: int = 200,
        mode_max: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        /**
         * Compute steady-state complex displacement field W(x,y) and
         * a normalized nodal-likelihood map Lw(x,y).
         *
         * @param freq       Driving frequency in Hz.
         * @param F0         Forcing amplitude in N.
         * @param x0         x-coordinate of force application (m).
         * @param y0         y-coordinate of force application (m).
         * @param num_points Grid resolution (points per side).
         * @param mode_max   Maximum modal index m,n.
         * @return X         2D array of x-coordinates (shape: num_points×num_points).
         * @return Y         2D array of y-coordinates (shape: num_points×num_points).
         * @return W         Complex displacement field (shape: num_points×num_points).
         * @return Lw        Normalized nodal-likelihood map ∈ [0,1].
         */
        """
        # Store drive parameters
        self.freq: float = freq
        self.F0: float = F0
        self.x0: float = x0
        self.y0: float = y0
        self.omega: float = 2 * np.pi * self.freq

        # Create uniform grid
        x = np.linspace(0, self.a, num_points)
        y = np.linspace(0, self.b, num_points)
        X, Y = np.meshgrid(x, y)

        # Modal superposition for complex displacement W
        W = np.zeros_like(X, dtype=complex)
        for m in range(1, mode_max + 1):
            for n in range(1, mode_max + 1):
                Phi_mn = (
                    np.sin(m * np.pi * X / self.a) *
                    np.sin(n * np.pi * Y / self.b)
                )
                omega_mn = np.sqrt(
                    (self.D / (self.rho * self.h)) *
                    ((m * np.pi / self.a)**2 + (n * np.pi / self.b)**2)**2
                )
                denom = (
                    self.rho * self.h *
                    (omega_mn**2 - self.omega**2 +
                     1j * 2 * self.zeta * omega_mn * self.omega)
                )
                Phi0 = (
                    np.sin(m * np.pi * x0 / self.a) *
                    np.sin(n * np.pi * y0 / self.b)
                )
                W += (F0 * Phi0 / denom) * Phi_mn

        # Build nodal-likelihood from real part of W
        amp = np.abs(W.real)
        amin, amax = amp.min(), amp.max()
        likelihood = 1.0 - (amp - amin) / (amax - amin)

        # Edge‐taper mask to downweight boundary effects
        nx, ny = likelihood.shape
        wx = 1 - np.abs(2 * np.linspace(0, 1, nx) - 1)
        wy = 1 - np.abs(2 * np.linspace(0, 1, ny) - 1)
        mask = np.outer(wx, wy)

        Lw = likelihood * mask
        Lw = (Lw - Lw.min()) / (Lw.max() - Lw.min())

        return X, Y, W, Lw
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- User‐adjustable parameters ---
    freq       = 2100             # drive frequency in Hz
    F0         = 1.0            # forcing amplitude in N
    x0, y0     = 0.12, 0.12      # driver location (m)
    num_points = 300             # grid resolution
    mode_max   = 20               # include modes 1..8

    # Plate properties (match your simulation parameters)
    a, b   = 0.24, 0.24          # plate dimensions (m)
    h      = 0.0005              # thickness (m)
    rho    = 7850                # density (kg/m³)
    E      = 200e9               # Young’s modulus (Pa)
    nu     = 0.3                 # Poisson’s ratio
    zeta   = 0.01                # modal damping ratio

    # Instantiate and compute
    plate = ChladniPlate(a, b, h, rho, E, nu, zeta)
    X, Y, W, Lw = plate.compute_contours(
        freq, F0, x0, y0,
        num_points=num_points,
        mode_max=mode_max
    )

    # Plot nodal lines (zero‐contour of the real part)
    plt.figure(figsize=(6,6))
    plt.contour(X, Y, W.real, levels=[0], linewidths=1.5, colors='black')
    plt.title(f"Chladni Nodal Lines at {freq} Hz")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
