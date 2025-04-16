#!/usr/bin/env python
"""
Realistic Particle Simulation of Sand on Chladni Plate (CPU Version):
- Uses correct sand grain mass (~11 mg)
- High damping to simulate surface friction
- CPU-based particle updates and repulsion
"""

import pygame
import pickle
import numpy as np
import sys
from scipy.ndimage import map_coordinates
import matplotlib.colors as mcolors

##############################
# 1. Load PDE Data and Precompute Gradients
##############################

with open("pde_solution.pkl", "rb") as f:
    data = pickle.load(f)

U_t = data['U_t']
x = data['x']
y = data['y']
t_array = data['t']
L = data['L']

num_t, num_points_y, num_points_x = U_t.shape
if num_points_y != num_points_x:
    raise ValueError("Grid must be square.")
num_points = num_points_x

U_min = np.min(U_t)
U_max = np.max(U_t)

# Precompute gradients
grad_x_all = np.empty_like(U_t)
grad_y_all = np.empty_like(U_t)
for i in range(num_t):
    dU_dy, dU_dx = np.gradient(U_t[i], y, x)
    grad_x_all[i] = dU_dx
    grad_y_all[i] = dU_dy

def fast_gradient(particles, sim_time):
    t_idx = np.searchsorted(t_array, sim_time)
    if t_idx >= len(t_array):
        t_idx = len(t_array) - 1
    x_norm = particles[:, 0] * (num_points - 1) / L
    y_norm = particles[:, 1] * (num_points - 1) / L
    g_x = map_coordinates(grad_x_all[t_idx], [y_norm, x_norm], order=1)
    g_y = map_coordinates(grad_y_all[t_idx], [y_norm, x_norm], order=1)
    return g_x, g_y

##############################
# 2. Particle Simulation Setup
##############################

num_particles = 5000
mass = 1.1e-5  # Sand grain mass in kg
damping_particle = 50
particle_radius = 0.0005
repulsion_strength = 1e-5

dt_orig = t_array[1] - t_array[0]
substeps = 1
dt_sub = dt_orig / substeps
T_total = t_array[-1]

np.random.seed(42)
particles = np.random.uniform(0, L, size=(num_particles, 2)).astype(np.float32)
velocities = np.zeros_like(particles, dtype=np.float32)

simulation_time = 0.0
max_speed = 0.1

cell_size = 2 * particle_radius
num_cells = int(np.ceil(L / cell_size))
cell_capacity = 20

##############################
# 3. CPU Collision Detection
##############################

def update_cell_lists(particles):
    cell_counts = np.zeros((num_cells, num_cells), dtype=np.int32)
    particle_order = -np.ones((num_cells * num_cells * cell_capacity,), dtype=np.int32)
    cell_indices = -np.ones((num_particles,), dtype=np.int32)

    for i in range(num_particles):
        cell_x = int(particles[i, 0] / cell_size)
        cell_y = int(particles[i, 1] / cell_size)
        if 0 <= cell_x < num_cells and 0 <= cell_y < num_cells:
            cell_idx = cell_y * num_cells + cell_x
            if cell_counts[cell_y, cell_x] < cell_capacity:
                slot = cell_counts[cell_y, cell_x]
                particle_order[cell_idx * cell_capacity + slot] = i
                cell_counts[cell_y, cell_x] += 1
                cell_indices[i] = cell_idx
    return cell_counts, cell_indices, particle_order

def compute_repulsion(particles, cell_counts, cell_indices, particle_order):
    rep_forces = np.zeros_like(particles, dtype=np.float32)

    for idx in range(num_particles):
        pos = particles[idx]
        cell_x = int(pos[0] / cell_size)
        cell_y = int(pos[1] / cell_size)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx = cell_x + dx
                ny = cell_y + dy
                if 0 <= nx < num_cells and 0 <= ny < num_cells:
                    cell_idx = ny * num_cells + nx
                    count = cell_counts[ny, nx]
                    for i in range(count):
                        p_idx = particle_order[cell_idx * cell_capacity + i]
                        if p_idx == idx or p_idx == -1:
                            continue
                        other_pos = particles[p_idx]
                        delta = pos - other_pos
                        dist = np.linalg.norm(delta)
                        if 1e-6 < dist < 2 * particle_radius:
                            force = repulsion_strength / (dist ** 2)
                            rep_forces[idx] += force * delta / dist
    return rep_forces

##############################
# 4. Pygame Setup
##############################

pygame.init()
window_size = (600, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Realistic Sand Simulation on Chladni Plate (CPU)")
clock = pygame.time.Clock()

font = pygame.font.SysFont('Arial', 16)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("blue_orange", ["blue", "orange"])
background_cache = {}

def get_background_surface(sim_time):
    t_idx = int(sim_time / dt_orig)
    if t_idx not in background_cache:
        effective_time = min(sim_time, T_total)
        idx = np.searchsorted(t_array, effective_time)
        if idx >= len(t_array):
            idx = len(t_array) - 1
        U_frame = U_t[idx]
        U_norm = (U_frame - U_min) / (U_max - U_min) if U_max > U_min else np.zeros_like(U_frame)
        U_rgba = custom_cmap(U_norm)
        U_rgb = (U_rgba[..., :3] * 255).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.transpose(U_rgb, (1, 0, 2)))
        surface = pygame.transform.scale(surface, window_size)
        background_cache[t_idx] = surface
    return background_cache[t_idx]

##############################
# 5. Main Simulation Loop
##############################

running = True
scale = window_size[0] / L
particle_surface = pygame.Surface(window_size, pygame.SRCALPHA)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    for _ in range(substeps):
        effective_time = min(simulation_time, T_total)
        cell_counts, cell_indices, particle_order = update_cell_lists(particles)
        g_x_vals, g_y_vals = fast_gradient(particles, effective_time)
        grad_forces = np.stack((-g_x_vals, -g_y_vals), axis=-1)
        repulsion_forces = compute_repulsion(particles, cell_counts, cell_indices, particle_order)
        total_acc = (grad_forces + repulsion_forces) / mass - damping_particle * velocities
        velocities += dt_sub * total_acc

        # Speed limiting
        speeds = np.linalg.norm(velocities, axis=1)
        too_fast = speeds > max_speed
        velocities[too_fast] *= (max_speed / speeds[too_fast])[:, None]

        particles += velocities * dt_sub

        # Boundary conditions
        for i in range(2):
            out_of_bounds_low = particles[:, i] < 0
            out_of_bounds_high = particles[:, i] > L
            velocities[out_of_bounds_low | out_of_bounds_high, i] *= -1
            particles[out_of_bounds_low, i] *= -1
            particles[out_of_bounds_high, i] = 2 * L - particles[out_of_bounds_high, i]

        simulation_time += dt_sub

    # Render
    particle_surface.fill((0, 0, 0, 0))
    particle_pixels = (particles * scale).astype(np.int32)
    for pos in particle_pixels:
        if 0 <= pos[0] < window_size[0] and 0 <= pos[1] < window_size[1]:
            pygame.draw.circle(particle_surface, (0, 0, 0, 255), (pos[0], pos[1]), 2)

    screen.blit(get_background_surface(simulation_time), (0, 0))
    screen.blit(particle_surface, (0, 0))

    time_text = font.render(f"Time = {simulation_time:.2f} s", True, (0, 0, 0))
    screen.blit(time_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
