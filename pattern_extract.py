import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class PatternExtract:
    def __init__(
        self, image_path: str, output_path: str, sigma: float = 10.0,
        cmap: str = 'Blues_r', figsize: tuple = (6, 6),
        interpolation: str = 'bicubic', dpi: int = 300
    ):
        self.image_path = image_path
        self.sigma = sigma
        self.cmap = cmap
        self.figsize = figsize
        self.interpolation = interpolation
        self.dpi = dpi
        self.output_path = output_path
        self.heat = None
        self.smoothed = None

    def _load_image(self) -> np.ndarray:
        return cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

    def _compute_heatmap(self, img: np.ndarray) -> np.ndarray:
        inv = 255 - img
        heat = inv.astype(np.float32)
        heat = (heat - heat.min()) / (heat.max() - heat.min())
        self.heat = heat
        return heat

    def _smooth_heatmap(self) -> np.ndarray:
        self.smoothed = gaussian_filter(self.heat, sigma=self.sigma)
        return self.smoothed

    def _save_heatmap(self):
        plt.figure(figsize=self.figsize)
        plt.imshow(
            self.smoothed,
            origin='lower',
            cmap=self.cmap,
            interpolation=self.interpolation
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def run(self):
        img = self._load_image()
        self._compute_heatmap(img)
        self._smooth_heatmap()
        self._save_heatmap()


if __name__ == '__main__':
    extractor = PatternExtract(
        image_path='lab_dataset/data_6.jpg',
        output_path='smoothed_heatmap.png',
        sigma=10.0,
    )
    extractor.run()
    print(f"Saved heatmap to {extractor.output_path}")
