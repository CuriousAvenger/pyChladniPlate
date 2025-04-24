import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from chladni_plate import ChladniPlate

class PatternPredict:
    def __init__(
        self, base_params: dict, scales: list,
        freq_range: tuple, target_path: str,
    ):
        self.base_params = base_params
        self.scales = scales
        self.f_min, self.f_max, self.f_step = freq_range
        self.target_path = target_path

    @staticmethod
    def load_and_normalize(path: str, nx: int, ny: int) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load '{path}'")
        resized = cv2.resize(img, (nx, ny), interpolation=cv2.INTER_AREA)
        f = resized.astype(np.float32)
        return (f - f.min()) / (f.max() - f.min())

    def simulate_pattern(self, freq: float, params: dict) -> np.ndarray:
        # Initialize plate (x0,y0 default to center if missing)
        plate = ChladniPlate(
            a=params['a'], b=params['b'],
            h=params['h'], rho=params['rho'],
            E=params['E'], nu=params['nu'],
            F0=params['F0'], c_damp=params['c_damp'],
            m_max=params['m_max'], n_max=params['n_max'],
            Nx=params['Nx'], Ny=params['Ny'],
            x0=params.get('x0'), y0=params.get('y0'),
        )
        Z = plate.compute_response(freq)
        return plate.compute_intensity(Z)

    def find_best_match(self) -> dict:
        best = {'ssim': -1.0, 'frequency': None, 'scale': None}
        freqs = np.arange(self.f_min, self.f_max + self.f_step, self.f_step)

        for scale in self.scales:
            params = self.base_params.copy()
            params['a'] = self.base_params['a'] * scale
            params['b'] = self.base_params['b'] * scale
            if 'x0' in self.base_params:
                params['x0'] = self.base_params['x0'] * scale
            if 'y0' in self.base_params:
                params['y0'] = self.base_params['y0'] * scale

            target = self.load_and_normalize(
                self.target_path,
                params['Nx'], params['Ny']
            )

            print(f"Scanning scale={scale:.2f} over {len(freqs)} frequencies...")
            for f in freqs:
                pred = self.simulate_pattern(f, params)
                score = ssim(target, pred, data_range=1.0)
                if score > best['ssim']:
                    best.update({'ssim': score, 'frequency': f, 'scale': scale})
            print(
                f" → scale={scale:.2f} best: {best['frequency']:.1f} Hz "
                f"(SSIM={best['ssim']:.4f})"
            )

        return best

if __name__ == '__main__':
    base_params = {
        'a': 0.24, 'b': 0.24, 'h': 0.0008,
        'rho': 3125, 'E': 1e9, 'nu': 0.3,
        'F0': 1.0, 'c_damp': 5.0,
        'm_max': 10, 'n_max': 10,
        'Nx': 200, 'Ny': 200,
    }
    scales = [1.0, 0.8, 0.6, 0.4, 0.2]
    freq_range = (0.0, 300.0, 1.0)
    target_path = 'test_images/image2.png'

    matcher = PatternPredict(base_params, scales, freq_range, target_path)
    result = matcher.find_best_match()
    print("\n🏆 Overall best match:")
    print(f"  Frequency:    {result['frequency']:.1f} Hz")
    print(f"  Scale factor: {result['scale']:.2f}")
    print(f"  SSIM score:   {result['ssim']:.4f}")
