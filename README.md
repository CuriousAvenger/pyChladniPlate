# pyChladniPlate

pyChladniPlate is a Python library for modeling and analyzing Chladni plate vibration patterns through a combination of theoretical simulation and image-based experimental matching. It provides a **ChladniPlate** class that computes steady-state displacement fields and normalized nodal-likelihood maps via modal superposition on a simply-supported rectangular plate ([GitHub][1]), and a **ChladniPredict** class that extracts skeletonized nodal lines from experimental images, maps them to physical plate coordinates across a range of scales, and sweeps driving frequencies to identify the best match ([GitHub][2]).

## Features

* **Modal superposition simulation**
  Compute the complex displacement field $W(x,y)$ and a normalized nodal-likelihood map $L_w(x,y)$ for a driven rectangular plate using up to `mode_max` modes.
* **Image preprocessing & skeletonization**
  Denoise, morphologically clean, and skeletonize experimental images using OpenCV and scikit-image.
* **Physical coordinate mapping**
  Convert pixel skeleton coordinates to physical plate coordinates centered on the plate, with adjustable scaling.
* **Automated parameter sweep**
  Sweep over user-provided frequency and scale arrays to compute error metrics (mean distance, percentage error, contour size adjustment) and identify the optimal (scale, frequency) combination.
* **Visualization tools**
  Plot adjusted error vs frequency for each scale, display the raw image, overlay skeleton points on normalized grayscale, and compare skeleton vs theoretical nodal lines in physical units.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/CuriousAvenger/pyChladniPlate.git
cd pyChladniPlate
pip install numpy opencv-python scikit-image scipy matplotlib
```

* **NumPy** – fundamental package for array computing
* **OpenCV** – image operations and morphological filtering
* **scikit-image** – skeletonization and contour finding
* **SciPy** – efficient KD-tree nearest-neighbor searches with cKDTree
* **Matplotlib** – plotting and visualization

## Usage

A simple example using **main.py**:

```python
#!/usr/bin/env python3
import numpy as np
from chladni_plate import ChladniPlate
from chladni_predict import ChladniPredict

if __name__ == "__main__":
    # Initialize plate (0.24 m × 0.24 m, 0.5 mm thick, steel properties)
    plate = ChladniPlate(a=0.24, b=0.24, h=0.0005,
                         rho=7850, E=200e9, nu=0.3, zeta=0.01)
    # Define search ranges
    freqs  = np.arange(200, 400, 100)
    scales = np.arange(0.8, 1.2, 0.2)
    img_path = "lab_dataset/data_7.jpg"

    print(f"Processing {img_path}...")
    predictor = ChladniPredict(plate, img_path, freqs, scales)
    best_scale, best_freq, best_err = predictor.run()
    print(f"Best: scale={best_scale:.2f}, freq={best_freq} Hz, err={best_err:.2f}%")
    predictor.plot_error_analysis()
    predictor.visualize()
```
## API Reference

### `ChladniPlate(a, b, h, rho, E, nu, zeta)`

* **Parameters & Attributes**

  * `a`, `b` (float): Plate width and height in meters.
  * `h` (float): Plate thickness in meters.
  * `rho` (float): Material density in kg/m³.
  * `E` (float): Young’s modulus in Pa.
  * `nu` (float): Poisson’s ratio (dimensionless).
  * `zeta` (float): Modal damping ratio (dimensionless).
  * `D` (float): Flexural rigidity $E h^3 / [12(1-\nu^2)]$.
  
* **Methods**

  ```python
  compute_contours(freq, F0, x0, y0, num_points=200, mode_max=20)
  ```

  * Computes grid coordinates `(X, Y)`, complex field `W`, and normalized nodal-likelihood `Lw`.

### `ChladniPredict(plate, img_path, freqs, scales, F0=1.0)`

* **Parameters**

  * `plate` (`ChladniPlate`): Initialized plate instance.
  * `img_path` (str): Path to experimental image.
  * `freqs`, `scales` (`np.ndarray`): Arrays of frequencies (Hz) and scale factors to test.
  * `F0` (float): Driving force amplitude (default 1.0 N).
* **Key Methods**

  ```python
  run() -> (best_scale, best_freq, best_error)
  plot_error_analysis()
  visualize()
  ```

  * Executes the full pipeline (skeleton extraction, frequency/scale sweep, error adjustment) and returns the optimal parameters.
  * Plots adjusted error vs frequency for each tested scale.
  * Displays: (1) raw RGB image, (2) normalized grayscale + skeleton overlay, (3) physical overlay of skeleton vs theoretical contours at best frequency.  

## Contributing

Feel free to open issues or submit pull requests to add features, fix bugs, or improve documentation. This project is licensed under **GPL-3.0**.

