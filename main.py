#!/usr/bin/env python3
import numpy as np

from chladni_plate import ChladniPlate
from chladni_predict import ChladniPredict

if __name__ == "__main__":
    plate   = ChladniPlate(a=0.24, b=0.24, h=0.0005,rho=7850, E=200e9, nu=0.3, zeta=0.01)
    freqs   = np.arange(200, 1800, 10)
    scales  = np.arange(0.8, 0.81, 0.2)

    for i in range(1, 16):
        img_path = f"lab_dataset/data_{i}.jpg"
        print(f"Processing {img_path}...")
        predictor = ChladniPredict(plate, img_path, freqs, scales)
        best_scale, best_freq, best_err = predictor.run()
        print(f"Best: scale={best_scale:.2f}, freq={best_freq} Hz, err={best_err:.2f}%")

        # predictor.plot_error_analysis()
        predictor.visualize()

    

