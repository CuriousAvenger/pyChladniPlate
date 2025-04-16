#!/usr/bin/env python
"""
Chladni Plate Vibration Simulation

This module simulates the vibration of a square Chladni plate, where the plate’s vibration
is excited by a point source approximated using a Dirac delta function. The simulation
solves the 2D plate PDE using scipy.integrate.odeint. Snapshots of the plate displacement
are displayed and the complete solution is saved into a pickle file.

Classes:
    Grid: Encapsulates grid generation and related parameters.
    ChladniPlateSimulator: Holds physical parameters, constructs the PDE system, performs integration,
                           and handles plotting and saving of data.
                           
Usage:
    from chladni_plate_simulation import ChladniPlateSimulator
    simulator = ChladniPlateSimulator()
    simulator.run_simulation()     # Integrates the PDE and displays integration time
    simulator.plot_snapshots()     # Plots snapshots of the simulation at selected times
    simulator.save_solution()        # Saves the resulting data to 'pde_solution.pkl'
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle
import time

class Grid:
    """
    Handles the grid setup for the simulation.
    """
    def __init__(self, L: float, num_points: int):
        """
        Initialize a grid for a square plate.
        Args:
            L (float): Side length of the plate.
            num_points (int): Number of grid points per side.
        """
        self.L = L
        self.num_points = num_points
        self.x = np.linspace(0, L, num_points)
        self.y = np.linspace(0, L, num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

class ChladniPlateSimulator:
    """
    Simulates the vibration of a Chladni plate.
    """
    def __init__(self,
                 L: float = 0.34,
                 nu: float = 0.5,
                 gamma: float = 0.015,
                 s0: float = 0.008,
                 m: int = 7,
                 n: int = 9,
                 num_points: int = 50,
                 t_final: float = 2.0,
                 fps: int = 30):
        """
        Initialize simulation parameters.
        Args:
            L (float): Plate side length in meters.
            nu (float): Transverse wave speed (m/s).
            gamma (float): Damping constant (1/s).
            s0 (float): Amplitude of the driving source.
            m (int): Mode shape parameter (number of nodal lines in x).
            n (int): Mode shape parameter (number of nodal lines in y).
            num_points (int): Grid resolution per side.
            t_final (float): Final simulation time in seconds.
            fps (int): Frames per second for the simulation.
        """
        # Physical properties
        self.L = L
        self.nu = nu
        self.gamma = gamma
        self.s0 = s0
        self.m = m
        self.n = n
        
        # Compute resonant frequency and angular frequency
        self.frequency = self.resonant_frequency(m, n, L, nu)
        self.omega = 2 * np.pi * self.frequency
        
        # Source location: center of plate
        self.x_source = L / 2
        self.y_source = L / 2
        
        # Grid setup
        self.grid = Grid(L, num_points)
        self.num_points = num_points
        
        # Time setup
        self.t_final = t_final
        self.fps = fps
        self.frames = int(t_final * fps)
        self.t = np.linspace(0, t_final, self.frames)
        
        # Initial conditions: plate at rest
        self.U0 = np.zeros((num_points, num_points))
        self.V0 = np.zeros((num_points, num_points))
        self.y0 = np.concatenate([self.U0.flatten(), self.V0.flatten()])
        
        # Placeholder for PDE solution
        self.U_t = None
        self.print_simulation_info()

    @staticmethod
    def resonant_frequency(m: int, n: int, L: float, nu: float) -> float:
        """
        Compute the resonant frequency for given mode shape and physical properties.
        Args:
            m (int): Nodal lines in x.
            n (int): Nodal lines in y.
            L (float): Plate side length.
            nu (float): Wave speed.
        Returns: float: Resonant frequency in Hz.
        """
        return (nu / 2) * np.sqrt((m / L) ** 2 + (n / L) ** 2)

    def print_simulation_info(self):
        """Prints the simulation configuration."""
        dx = self.grid.dx
        print("========== Chladni Plate Vibration Simulation ==========")
        print(f" Plate size      : {self.L} m x {self.L} m (square)")
        print(f" Grid resolution : {self.num_points} x {self.num_points} (spacing {dx:.5f} m)")
        print(f" Time range      : 0 to {self.t_final} s ({self.frames} frames at {self.fps} FPS)")
        print(f" Damping gamma   : {self.gamma} 1/s")
        print(f" Wave speed      : {self.nu} m/s")
        print(f" Source amplitude: {self.s0}")
        print(f" Mode shape      : m = {self.m}, n = {self.n}")
        print(f" Resonant freq   : {self.frequency:.2f} Hz ({self.omega:.2f} rad/s)")
        print(f" Source location : x = {self.x_source} m, y = {self.y_source} m")
        print("=======================================================\n")

    def dirac_delta(self, x: np.ndarray, x0: float, epsilon: float = None) -> np.ndarray:
        """
        Approximates a Dirac delta function using a narrow Gaussian.
        Args: 
            x (np.ndarray): Spatial coordinate.
            x0 (float): Location of the delta impulse.
            epsilon (float): Width of the Gaussian (default: grid spacing in x).
        Returns: np.ndarray: Approximated delta function.
        """
        if epsilon is None:
            epsilon = self.grid.dx
        return np.exp(-((x - x0) ** 2) / (2 * epsilon ** 2)) / (epsilon * np.sqrt(2 * np.pi))

    def source_function(self, t: float) -> np.ndarray:
        """
        Computes the source term at time t over the grid.
        Args: t (float): Time instant.
        Returns: np.ndarray: Source term S(X,Y,t) as a 2D array.
        """
        delta_x = self.dirac_delta(self.grid.X, self.x_source)
        delta_y = self.dirac_delta(self.grid.Y, self.y_source)
        return self.s0 * delta_x * delta_y * np.sin(self.omega * t)

    def laplacian(self, U: np.ndarray) -> np.ndarray:
        """
        Computes the 2D Laplacian of U with simple Neumann boundary approximations.
        Args: U (np.ndarray): 2D displacement field.
        Returns: np.ndarray: Laplacian of U.
        """
        lap = np.zeros_like(U)
        dx, dy = self.grid.dx, self.grid.dy
        lap[1:-1, 1:-1] = (
            (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[0:-2, 1:-1]) / dx**2 +
            (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, 0:-2]) / dy**2
        )
        # Neumann boundary conditions: mirror second row/column
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]
        return lap

    def pde_system(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Defines the PDE system for the plate's vibration.
        Args:
            y (np.ndarray): Flattened state vector [U, V].
            t (float): Time.
        Returns:
            np.ndarray: Time derivative of the state vector.
        """
        num_pts = self.num_points ** 2
        U = y[:num_pts].reshape((self.num_points, self.num_points))
        V = y[num_pts:].reshape((self.num_points, self.num_points))
        dUdt = V
        lap = self.laplacian(U)
        S = self.source_function(t)
        dVdt = (self.nu ** 2) * lap - self.gamma * V + S
        return np.concatenate([dUdt.flatten(), dVdt.flatten()])

    def run_simulation(self):
        """
        Solves the PDE using odeint and stores the displacement field U over time.
        """
        print("[INFO] Starting PDE integration...")
        start_time = time.perf_counter()
        solution = odeint(self.pde_system, self.y0, self.t)
        end_time = time.perf_counter()
        print(f"[INFO] PDE integration completed in {end_time - start_time:.4f} seconds")

        # Extract displacement field U over time
        num_pts = self.num_points ** 2
        self.U_t = solution[:, :num_pts].reshape((len(self.t), self.num_points, self.num_points)).astype(np.float32)

    def plot_snapshots(self):
        """
        Plots snapshots of the displacement field at selected times.
        """
        if self.U_t is None:
            raise RuntimeError("Simulation data not available. Run run_simulation() first.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        # Choose three time steps: beginning, middle, and near the end.
        time_steps = [0, int(self.frames / 3), int(2 * self.frames / 3)]
        for i, ax in zip(time_steps, axes):
            ax.set_title(f"t = {self.t[i]:.2f} s (mode m={self.m}, n={self.n})")
            contour = ax.contourf(self.grid.X, self.grid.Y, self.U_t[i], levels=100, cmap='RdBu_r')
            ax.set_axis_off()
        fig.colorbar(contour, ax=axes, orientation='vertical')
        plt.show()

    def save_solution(self, filename: str = "pde_solution.pkl"):
        """
        Saves the PDE solution and grid parameters to a pickle file.
        Args:
            filename (str): Name of the file to save the solution.
        """
        if self.U_t is None:
            raise RuntimeError("No solution available to save. Run run_simulation() first.")

        data = {
            'U_t': self.U_t,
            'x': self.grid.x,
            'y': self.grid.y,
            't': self.t,
            'L': self.L
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"[INFO] PDE solution saved to '{filename}'.")


def main():
    simulator = ChladniPlateSimulator()
    simulator.run_simulation()
    simulator.plot_snapshots()
    simulator.save_solution()


if __name__ == "__main__":
    main()
