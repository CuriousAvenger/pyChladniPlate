import numpy as np
import matplotlib.pyplot as plt

class ChladniPlate:
    def __init__(
        self, a: float, b: float, h: float,
        rho: float, E: float, nu: float,
        F0: float, c_damp: float, m_max: int,
        n_max: int, Nx: int, Ny: int,
        x0: float = None, y0: float = None,
    ):
        self.a, self.b, self.h = a, b, h
        self.rho, self.E, self.nu = rho, E, nu
        self.F0, self.c_damp = F0, c_damp
        self.m_max, self.n_max = m_max, n_max
        self.Nx, self.Ny = Nx, Ny
        self.x0 = x0 if x0 is not None else a / 2
        self.y0 = y0 if y0 is not None else b / 2

        self._compute_derived_constants()
        self._compute_modal_coefficients()
        self._compute_modal_shapes()

    def _compute_derived_constants(self):
        self.rho_h = self.rho * self.h
        self.D = self.E * self.h**3 / (12 * (1 - self.nu**2))
        self.scale = np.sqrt(self.D / self.rho_h)

    def _compute_modal_coefficients(self):
        omegas, phi0 = [], []
        for m in range(1, self.m_max + 1):
            for n in range(1, self.n_max + 1):
                k2 = (m / self.a)**2 + (n / self.b)**2
                omega_mn = np.pi**2 * k2 * self.scale
                omegas.append(omega_mn)
                phi0.append(
                    np.sin(m * np.pi * self.x0 / self.a) *
                    np.sin(n * np.pi * self.y0 / self.b)
                )
        self.omegas = np.array(omegas)
        self.phi0 = np.array(phi0)

    def _compute_modal_shapes(self):
        x = np.linspace(0, self.a, self.Nx)
        y = np.linspace(0, self.b, self.Ny)
        self.xx, self.yy = np.meshgrid(x, y)
        phis = []
        for m in range(1, self.m_max + 1):
            for n in range(1, self.n_max + 1):
                phis.append(
                    np.sin(m * np.pi * self.xx / self.a) *
                    np.sin(n * np.pi * self.yy / self.b)
                )
        self.phis = np.stack(phis, axis=0)

    def compute_response(self, freq: float) -> np.ndarray:
        omega_drive = 2 * np.pi * freq
        denom = self.rho_h * (self.omegas**2 - omega_drive**2) + self.c_damp * omega_drive
        A = self.F0 * self.phi0 / denom
        return np.tensordot(A, self.phis, axes=(0, 0))

    def compute_intensity(self, Z: np.ndarray) -> np.ndarray:
        Znorm = np.abs(Z)
        Znorm /= Znorm.max()
        dist_edge = np.minimum.reduce([
            self.xx, self.a - self.xx,
            self.yy, self.b - self.yy
        ])
        w = np.clip(1 - (dist_edge * 2 / min(self.a, self.b)), 0, 1)
        return np.clip(Znorm * (1 - w) + w, 0, 1)
    

if __name__ == '__main__':
    plate = ChladniPlate(
        a=0.24, b=0.24,
        h=0.0008, rho=3125,
        E=1e9, nu=0.3,
        F0=1.0, c_damp=5.0,
        m_max=10, n_max=10,
        Nx=200, Ny=200,
    )
    freq = 77.0
    Z = plate.compute_response(freq)
    I = plate.compute_intensity(Z)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        I,
        origin='lower',
        extent=(0, plate.a, 0, plate.b),
        cmap='Blues_r',
        aspect='equal',
        interpolation='bicubic'
    )
    plt.colorbar(im, ax=ax, label='Normalized Intensity')
    ax.set_title(f"Chladni Plate — {freq:.1f} Hz")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.show()

