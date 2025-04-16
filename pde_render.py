#!/usr/bin/env python
"""
Realistic Particle Simulation of Sand on a Chladni Plate
GPU-Accelerated with PyOpenCL

This simulation models sand particles moving on a vibrating Chladni plate.
Particles experience a damping force, gradient forces from the PDE solution,
and inter-particle repulsion to prevent stacking. GPU acceleration is provided
via PyOpenCL for efficient particle update calculations.

Classes:
    PDEData: Loads the precomputed PDE solution and provides gradient interpolation.
    ParticleSimulationGPU: Sets up particle properties, OpenCL kernel, collision detection,
                           and updates particle state on the GPU.
    Renderer: Initializes Pygame and handles drawing of the background and particles.
    SandSimulationApp: The main application class that integrates the simulation and rendering loops.

Usage:
    Run this script to start the GPU-accelerated simulation.
"""

import pygame
import pickle
import numpy as np
import sys
from scipy.ndimage import map_coordinates
import matplotlib.colors as mcolors
import pyopencl as cl
import pyopencl.array as cl_array

class PDEData:
    """
    Handles loading of the PDE solution and precomputing spatial gradients.

    Attributes:
        U_t (np.ndarray): 3D array of PDE displacement fields over time.
        x (np.ndarray): x-coordinates of the grid.
        y (np.ndarray): y-coordinates of the grid.
        t_array (np.ndarray): Array of time values.
        L (float): Plate side length.
        grad_x_all (np.ndarray): Precomputed x-gradients of U_t.
        grad_y_all (np.ndarray): Precomputed y-gradients of U_t.
        U_min (float): Minimum displacement value.
        U_max (float): Maximum displacement value.
        num_points (int): Number of grid points (assumed square).
    """

    def __init__(self, filename: str = "pde_solution.pkl"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.U_t = data['U_t']
        self.x = data['x']
        self.y = data['y']
        self.t_array = data['t']
        self.L = data['L']

        num_t, num_points_y, num_points_x = self.U_t.shape
        if num_points_y != num_points_x:
            raise ValueError("Grid must be square.")
        self.num_points = num_points_x

        self.U_min = np.min(self.U_t)
        self.U_max = np.max(self.U_t)

        self.grad_x_all = np.empty_like(self.U_t)
        self.grad_y_all = np.empty_like(self.U_t)
        self._precompute_gradients()

    def _precompute_gradients(self):
        """Precompute the spatial gradients of the PDE solution for fast interpolation."""
        for i in range(self.U_t.shape[0]):
            # Note: np.gradient returns (dU/dy, dU/dx)
            dU_dy, dU_dx = np.gradient(self.U_t[i], self.y, self.x)
            self.grad_x_all[i] = dU_dx
            self.grad_y_all[i] = dU_dy

    def fast_gradient(self, particles: np.ndarray, sim_time: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolates the precomputed gradients at the particle positions for a given simulation time.
        Args:
            particles (np.ndarray): Particle positions, shape (n_particles, 2).
            sim_time (float): Current simulation time.
        Returns: Tuple of np.ndarray: (g_x, g_y) gradient components at particle locations.
        """
        t_idx = np.searchsorted(self.t_array, sim_time)
        if t_idx >= len(self.t_array):
            t_idx = len(self.t_array) - 1
        x_norm = particles[:, 0] * (self.num_points - 1) / self.L
        y_norm = particles[:, 1] * (self.num_points - 1) / self.L
        g_x = map_coordinates(self.grad_x_all[t_idx], [y_norm, x_norm], order=1)
        g_y = map_coordinates(self.grad_y_all[t_idx], [y_norm, x_norm], order=1)
        return g_x, g_y

class ParticleSimulationGPU:
    """
    Manages the GPU-accelerated particle simulation using PyOpenCL.

    Attributes:
        num_particles (int): Number of particles in the simulation.
        mass (float): Mass of each sand particle (kg).
        damping_particle (float): Damping factor (friction behavior).
        particle_radius (float): Radius of each sand particle.
        repulsion_strength (float): Strength of repulsive force to prevent stacking.
        dt_sub (float): Simulation time-step.
        T_total (float): Total simulation time.
        particles (np.ndarray): Particle positions.
        velocities (np.ndarray): Particle velocities.
        cell_size (float): Size of a cell in spatial grid for collision detection.
        num_cells (int): Number of cells per dimension.
        cell_capacity (int): Maximum particles per cell.
        cell_counts (np.ndarray): For each cell, how many particles reside.
        cell_indices (np.ndarray): Mapping of particle to cell.
        particle_order (np.ndarray): Flat storage of particle indices by cell.
        context, queue, program: PyOpenCL objects.
        GPU buffers: particles_gpu, velocities_gpu, grad_x_gpu, grad_y_gpu, cell_counts_gpu, cell_indices_gpu, particle_order_gpu.
    """

    def __init__(self, L: float, t_array: np.ndarray, num_particles: int = 5000):
        # Simulation parameters
        self.L = L
        self.t_array = t_array
        self.num_particles = num_particles
        self.mass = 1.1e-5
        self.damping_particle = 50
        self.particle_radius = 0.0005
        self.repulsion_strength = 1e-5

        self.dt_orig = t_array[1] - t_array[0]
        self.substeps = 1
        self.dt_sub = self.dt_orig / self.substeps
        self.T_total = t_array[-1]

        np.random.seed(42)
        self.particles = np.random.uniform(0, L, size=(self.num_particles, 2)).astype(np.float32)
        self.velocities = np.zeros_like(self.particles, dtype=np.float32)
        self.simulation_time = 0.0
        self.max_speed = 0.1

        # Collision detection grid parameters
        self.cell_size = 2 * self.particle_radius
        self.num_cells = int(np.ceil(L / self.cell_size))
        self.cell_capacity = 20
        total_slots = self.num_cells * self.num_cells * self.cell_capacity
        self.cell_counts = np.zeros((self.num_cells, self.num_cells), dtype=np.int32)
        self.cell_indices = np.zeros((self.num_particles,), dtype=np.int32)
        self.particle_order = np.full((total_slots,), -1, dtype=np.int32)

        # Initialize PyOpenCL context and compile kernel
        self._init_opencl()

    def _init_opencl(self):
        """Sets up the OpenCL context, command queue, and compiles the simulation kernel."""
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)

        # Define and compile the OpenCL kernel
        kernel_code = """
        __kernel void update_particles(
            __global float2 *particles,
            __global float2 *velocities,
            __global float *grad_x,
            __global float *grad_y,
            __global int *cell_counts,
            __global int *cell_indices,
            __global int *particle_order,
            float dt,
            float mass,
            float damping,
            float L,
            float max_speed,
            float particle_radius,
            float repulsion_strength,
            float cell_size,
            int num_cells,
            int cell_capacity)
        {
            int idx = get_global_id(0);
            if (idx >= %d) return;

            // Compute gradient force
            float force_x = -grad_x[idx];
            float force_y = -grad_y[idx];

            // Compute repulsive forces
            float2 pos = particles[idx];
            int cell_x = (int)(pos.x / cell_size);
            int cell_y = (int)(pos.y / cell_size);

            float rep_force_x = 0.0f;
            float rep_force_y = 0.0f;

            // Check neighboring cells
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nx = cell_x + dx;
                    int ny = cell_y + dy;
                    if (nx < 0 || nx >= num_cells || ny < 0 || ny >= num_cells) continue;

                    int cell_idx = ny * num_cells + nx;
                    int count = cell_counts[cell_idx];
                    for (int i = 0; i < count; i++) {
                        int p_idx = particle_order[cell_idx * cell_capacity + i];
                        if (p_idx == idx) continue;
                        float2 other_pos = particles[p_idx];
                        float dx = pos.x - other_pos.x;
                        float dy = pos.y - other_pos.y;
                        float dist = sqrt(dx * dx + dy * dy);
                        if (dist < 2 * particle_radius && dist > 1e-6) {
                            float force = repulsion_strength / (dist * dist);
                            rep_force_x += force * dx / dist;
                            rep_force_y += force * dy / dist;
                        }
                    }
                }
            }

            // Update velocity
            float acc_x = (force_x + rep_force_x) / mass - damping * velocities[idx].x;
            float acc_y = (force_y + rep_force_y) / mass - damping * velocities[idx].y;
            velocities[idx].x += dt * acc_x;
            velocities[idx].y += dt * acc_y;

            // Cap speed
            float speed = sqrt(velocities[idx].x * velocities[idx].x + velocities[idx].y * velocities[idx].y);
            if (speed > max_speed) {
                velocities[idx].x *= max_speed / speed;
                velocities[idx].y *= max_speed / speed;
            }

            // Update position
            particles[idx].x += dt * velocities[idx].x;
            particles[idx].y += dt * velocities[idx].y;

            // Handle boundaries
            if (particles[idx].x < 0) {
                particles[idx].x = -particles[idx].x;
                velocities[idx].x = -velocities[idx].x;
            } else if (particles[idx].x > L) {
                particles[idx].x = 2 * L - particles[idx].x;
                velocities[idx].x = -velocities[idx].x;
            }
            if (particles[idx].y < 0) {
                particles[idx].y = -particles[idx].y;
                velocities[idx].y = -velocities[idx].y;
            } else if (particles[idx].y > L) {
                particles[idx].y = 2 * L - particles[idx].y;
                velocities[idx].y = -velocities[idx].y;
            }
        }
        """ % self.num_particles

        self.program = cl.Program(self.context, kernel_code).build()

        # Allocate GPU buffers
        self.particles_gpu = cl_array.to_device(self.queue, self.particles)
        self.velocities_gpu = cl_array.to_device(self.queue, self.velocities)
        self.grad_x_gpu = cl_array.Array(self.queue, (self.num_particles,), dtype=np.float32)
        self.grad_y_gpu = cl_array.Array(self.queue, (self.num_particles,), dtype=np.float32)
        self.cell_counts_gpu = cl_array.to_device(self.queue, self.cell_counts)
        self.cell_indices_gpu = cl_array.to_device(self.queue, self.cell_indices)
        self.particle_order_gpu = cl_array.to_device(self.queue, self.particle_order)

    def update_cell_lists(self):
        """
        Assigns particles to spatial cells for collision detection and updates GPU buffers.
        """
        self.cell_counts.fill(0)
        self.cell_indices.fill(-1)
        self.particle_order.fill(-1)

        for i in range(self.num_particles):
            cell_x = int(self.particles[i, 0] / self.cell_size)
            cell_y = int(self.particles[i, 1] / self.cell_size)
            if 0 <= cell_x < self.num_cells and 0 <= cell_y < self.num_cells:
                cell_idx = cell_y * self.num_cells + cell_x
                if self.cell_counts[cell_y, cell_x] < self.cell_capacity:
                    slot = self.cell_counts[cell_y, cell_x]
                    self.particle_order[cell_idx * self.cell_capacity + slot] = i
                    self.cell_counts[cell_y, cell_x] += 1
                    self.cell_indices[i] = cell_idx

        # Copy updated arrays to GPU buffers
        cl.enqueue_copy(self.queue, self.cell_counts_gpu.data, self.cell_counts)
        cl.enqueue_copy(self.queue, self.cell_indices_gpu.data, self.cell_indices)
        cl.enqueue_copy(self.queue, self.particle_order_gpu.data, self.particle_order)

    def update_particles(self, grad_x: np.ndarray, grad_y: np.ndarray):
        """
        Updates particle positions and velocities on the GPU for one substep.
        Args:
            grad_x (np.ndarray): Gradient values in x direction.
            grad_y (np.ndarray): Gradient values in y direction.
        """
        # Copy gradient values to GPU
        cl.enqueue_copy(self.queue, self.grad_x_gpu.data, grad_x.astype(np.float32))
        cl.enqueue_copy(self.queue, self.grad_y_gpu.data, grad_y.astype(np.float32))

        global_size = (self.num_particles,)
        self.program.update_particles(
            self.queue, global_size, None,
            self.particles_gpu.data, self.velocities_gpu.data,
            self.grad_x_gpu.data, self.grad_y_gpu.data,
            self.cell_counts_gpu.data, self.cell_indices_gpu.data,
            self.particle_order_gpu.data,
            np.float32(self.dt_sub), np.float32(self.mass),
            np.float32(self.damping_particle), np.float32(self.L),
            np.float32(self.max_speed), np.float32(self.particle_radius),
            np.float32(self.repulsion_strength), np.float32(self.cell_size),
            np.int32(self.num_cells), np.int32(self.cell_capacity)
        )

        # Retrieve updated particle positions for rendering in CPU
        self.particles = self.particles_gpu.get()
        self.queue.finish()

class Renderer:
    """
    Handles rendering with Pygame and the background based on the PDE data.

    Attributes:
        window_size (tuple): Dimensions for the Pygame window.
        screen (pygame.Surface): The display surface.
        font (pygame.Font): Font used for text rendering.
        custom_cmap (matplotlib.colors.Colormap): Custom colormap for the background.
        background_cache (dict): Cache for already generated background surfaces.
        scale (float): Factor to scale simulation coordinates to screen pixels.
    """
    
    def __init__(self, L: float, dt_orig: float, U_t: np.ndarray, t_array: np.ndarray):
        pygame.init()
        self.window_size = (600, 600)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Realistic Sand Simulation on Chladni Plate (GPU with Repulsion)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 16)
        self.custom_cmap = mcolors.LinearSegmentedColormap.from_list("blue_orange", ["blue", "orange"])
        self.background_cache = {}
        self.L = L
        self.dt_orig = dt_orig
        self.U_t = U_t
        self.t_array = t_array
        self.scale = self.window_size[0] / L

    def get_background_surface(self, sim_time: float) -> pygame.Surface:
        """
        Returns the background surface corresponding to the current simulation time.
        Args:
            sim_time (float): Current simulation time.
        Returns: pygame.Surface: The background image surface.
        """
        t_idx = int(sim_time / self.dt_orig)
        if t_idx not in self.background_cache:
            effective_time = min(sim_time, self.t_array[-1])
            idx = np.searchsorted(self.t_array, effective_time)
            if idx >= len(self.t_array):
                idx = len(self.t_array) - 1
            U_frame = self.U_t[idx]
            if np.ptp(U_frame) > 0:
                U_norm = (U_frame - np.min(self.U_t)) / (np.ptp(self.U_t))
            else:
                U_norm = np.zeros_like(U_frame)
            U_rgba = self.custom_cmap(U_norm)
            U_rgb = (U_rgba[..., :3] * 255).astype(np.uint8)
            surface = pygame.surfarray.make_surface(np.transpose(U_rgb, (1, 0, 2)))
            surface = pygame.transform.scale(surface, self.window_size)
            self.background_cache[t_idx] = surface
        return self.background_cache[t_idx]

    def render(self, sim_time: float, particles: np.ndarray):
        """
        Renders the current frame, including background and particles.
        Args:
            sim_time (float): Current simulation time.
            particles (np.ndarray): Particle positions to be drawn.
        """
        particle_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
        particle_surface.fill((0, 0, 0, 0))
        particle_pixels = (particles * self.scale).astype(np.int32)
        for pos in particle_pixels:
            if 0 <= pos[0] < self.window_size[0] and 0 <= pos[1] < self.window_size[1]:
                pygame.draw.circle(particle_surface, (0, 0, 0, 255), (pos[0], pos[1]), 2)

        self.screen.blit(self.get_background_surface(sim_time), (0, 0))
        self.screen.blit(particle_surface, (0, 0))
        time_text = self.font.render(f"Time = {sim_time:.2f} s", True, (0, 0, 0))
        self.screen.blit(time_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(30)

class SandSimulationApp:
    """
    The main application class that integrates the PDE data, GPU simulation,
    and rendering for the sand simulation on the Chladni plate.
    """

    def __init__(self):
        # Load PDE data and gradients
        self.pde_data = PDEData("pde_solution.pkl")
        # Initialize GPU simulation components
        self.simulation = ParticleSimulationGPU(L=self.pde_data.L, t_array=self.pde_data.t_array, num_particles=5000)
        # Initialize rendering system
        self.renderer = Renderer(L=self.pde_data.L, dt_orig=self.simulation.dt_orig,
                                 U_t=self.pde_data.U_t, t_array=self.pde_data.t_array)
        self.running = True

    def run(self):
        """
        Main simulation loop that updates particle states and renders frames.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Perform simulation substeps (could be increased for higher resolution)
            for _ in range(self.simulation.substeps):
                effective_time = min(self.simulation.simulation_time, self.simulation.T_total)
                # Update collision detection cells
                self.simulation.update_cell_lists()
                # Compute gradient forces from PDE data (on CPU)
                g_x, g_y = self.pde_data.fast_gradient(self.simulation.particles, effective_time)
                # Update particle state on GPU
                self.simulation.update_particles(g_x, g_y)
                # Increment simulation time
                self.simulation.simulation_time += self.simulation.dt_sub

            # Render the current frame
            self.renderer.render(self.simulation.simulation_time, self.simulation.particles)
        pygame.quit()
        sys.exit()

def main():
    app = SandSimulationApp()
    app.run()

if __name__ == "__main__":
    main()
