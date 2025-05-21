#!/usr/bin/env python3
import numpy as np
import cv2
from typing import Tuple, List
from skimage.morphology import skeletonize
from skimage.measure    import find_contours
from scipy.spatial      import cKDTree
from chladni_plate      import ChladniPlate
import matplotlib.pyplot as plt


class ChladniPredict:
    """
    /**
     * Predicts the optimal driving frequency and scale for an experimental Chladni plate image.
     *
     * Sweeps over provided scales and frequencies, computes error metrics against experimental nodal skeletons,
     * adjusts errors for contour size, and identifies the best (scale, frequency) combination.
     */
    """

    def __init__(
        self,
        plate: ChladniPlate,
        img_path: str,
        freqs: np.ndarray,
        scales: np.ndarray,
        F0: float = 1.0
    ):
        """
        /**
         * @param plate     Initialized ChladniPlate instance.
         * @param img_path  Path to experimental image.
         * @param freqs     1D array of frequencies (Hz) to test.
         * @param scales    1D array of scale factors to test.
         * @param F0        Driving force amplitude (N).
         */
        """
        self.plate     = plate
        self.a         = plate.a
        self.b         = plate.b
        self.F0        = F0
        self.freqs     = freqs
        self.scales    = scales
        self.img_path  = img_path
        self.results: List[Tuple[float, float, float, float, int, float]] = []

    def _extract_skeleton(self, im_gray: np.ndarray,
                          median_ksize: int = 5,
                          morph_size: int = 7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        /**
         * Denoises, morphologically filters, and skeletonizes the image.
         * @return mask, skeleton_image, pixel_coords
         */
        """
        denoised = cv2.medianBlur(im_gray, median_ksize)
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
        cleaned  = cv2.morphologyEx(denoised, cv2.MORPH_OPEN,  kernel)
        cleaned  = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, kernel)
        mask     = (cleaned > 127).astype(np.uint8) * 255
        ske_bool = skeletonize(mask > 127)
        ys, xs   = np.nonzero(ske_bool)
        coords   = np.column_stack((xs, ys))
        return mask, ske_bool.astype(np.uint8) * 255, coords

    def _map_to_physical(self, coords: np.ndarray,
                         scale: float,
                         px_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        /**
         * Converts pixel coordinates to physical plate coordinates,
         * centered and scaled about plate center.
         */
        """
        h_px, w_px = px_shape
        x = coords[:, 0] * (self.a / w_px)
        y = coords[:, 1] * (self.b / h_px)
        cx, cy = self.a / 2, self.b / 2
        x = (x - cx) * scale + cx
        y = (y - cy) * scale + cy
        return np.column_stack((x, y))

    def _sweep(self, phys_coords: np.ndarray, scale: float):
        """
        /**
         * Sweeps all frequencies for a given scale, records raw errors.
         */
        """
        records: List[Tuple[float, float, float, int]] = []
        for f in self.freqs:
            print(f"\t\tFrequency: {f} Hz")
            _, _, W, _ = self.plate.compute_contours(
                freq=f, F0=self.F0,
                x0=self.a/2, y0=self.b/2,
                num_points=200, mode_max=20
            )
            Z = W.real
            raw_contours = find_contours(Z, 0.0)
            if not raw_contours:
                records.append((f, np.nan, np.nan, 0))
                continue

            contour_rc = np.vstack(raw_contours)
            ny, nx = Z.shape
            dx, dy = self.a/(nx-1), self.b/(ny-1)
            contour_xy = np.column_stack([
                contour_rc[:, 1] * dx,
                contour_rc[:, 0] * dy
            ])
            tree = cKDTree(contour_xy)
            dists, _ = tree.query(phys_coords, k=1)
            mean_dist   = dists.mean()
            raw_pct_err = mean_dist / self.a * 100
            records.append((f, mean_dist, raw_pct_err, len(contour_xy)))

        Ms    = np.array([r[3] for r in records], dtype=float)
        M_med = np.nanmedian(Ms[Ms > 0])
        for f, md, rp, M in records:
            adj = rp * (M / M_med) if (M > 0 and not np.isnan(rp)) else np.nan
            self.results.append((scale, f, md, rp, M, adj))

    def run(self) -> Tuple[float, float, float]:
        """
        /**
         * Executes the full pipeline: load image, extract skeleton, sweep scales & freqs,
         * and determine best combination.
         * @return (best_scale, best_freq, best_adjusted_error)
         */
        """
        im_bgr  = cv2.imread(self.img_path)
        if im_bgr is None:
            raise FileNotFoundError(f"Cannot load '{self.img_path}'")
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        mask, _, pix_coords = self._extract_skeleton(im_gray)
        for scale in self.scales:
            print(f"\tScale: {scale}x")
            phys_coords = self._map_to_physical(pix_coords, scale, mask.shape)
            self._sweep(phys_coords, scale)

        valid = [(s, f, adj) for s, f, *_ , adj in self.results if not np.isnan(adj)]
        if not valid:
            return None, None, None
        best_scale, best_freq, best_err = min(valid, key=lambda x: x[2])
        return best_scale, best_freq, best_err

    def plot_error_analysis(self):
        """
        /**
         * Plots adjusted error vs frequency for each scale.
         */
        """
        plt.figure()
        for scale in self.scales:
            freqs = [r[1] for r in self.results if r[0] == scale]
            errs  = [r[5] for r in self.results if r[0] == scale]
            plt.plot(freqs, errs, label=f"scale={scale:.2f}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Adjusted Error (%)")
        plt.legend()
        plt.title("Error Analysis: Adjusted Error vs Frequency per Scale")
        plt.show()

    def visualize(self):
        """
        /**
         * Displays three separate figures:
         * 1) Raw RGB image.
         * 2) Normalized grayscale image with skeleton points overlaid in red.
         * 3) Physical overlay of skeleton vs theoretical nodal lines in meters.
         */
        """
        # Ensure results loaded
        if not self.results:
            raise RuntimeError("No results found. Please call run() before visualize().")

        # Determine best parameters
        valid = [(s, f, adj) for s, f, *_, adj in self.results if not np.isnan(adj)]
        best_scale, best_freq, _ = min(valid, key=lambda x: x[2])

        # Load images
        im_bgr = cv2.imread(self.img_path)
        if im_bgr is None:
            raise FileNotFoundError(f"Cannot load '{self.img_path}'")
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        im_rgb  = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        # Extract skeleton
        mask, skel, coords = self._extract_skeleton(im_gray)
        h_px, w_px = mask.shape

        # Map pixel to physical coords
        x_phys = coords[:, 0] * (self.a / w_px)
        y_phys = coords[:, 1] * (self.b / h_px)
        # Apply best scale
        cx, cy = self.a / 2, self.b / 2
        x_phys = (x_phys - cx) * best_scale + cx
        y_phys = (y_phys - cy) * best_scale + cy

        # Compute theoretical field at best frequency
        X, Y, W, _ = self.plate.compute_contours(
            freq=best_freq, F0=self.F0,
            x0=self.a/2, y0=self.b/2,
            num_points=200, mode_max=20
        )

        # Normalized + skeleton on gray
        plt.figure(figsize=(6,6))
        norm = (im_gray - im_gray.min()) / (im_gray.max() - im_gray.min())
        ys, xs = np.nonzero(skel)
        plt.imshow(norm, cmap='gray')
        plt.scatter(xs, ys, s=1, c='red')
        plt.title(f"Normalized + Skeleton (scale={best_scale:.2f})")
        plt.axis('off')

        # Physical overlay of skeleton and model
        plt.figure(figsize=(6,6))
        plt.scatter(x_phys, y_phys, s=1, c='black', label='Skeleton')
        plt.contour(X, Y, W.real, levels=[0], linewidths=1.5, colors='red', label='Nodal Lines')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Skeleton vs Model Contours @ {best_freq:.0f} Hz')
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()
        plt.show()
