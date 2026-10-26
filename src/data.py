"""Synthetic micrograph generator.

Each sample is a 2D grain microstructure (Voronoi tessellation) rendered as a
grayscale image. The hidden physical property is a Hall-Petch-style hardness
proxy, H = H0 + k / sqrt(d), where d is the mean grain diameter in micrometres.

Images are rendered at different magnifications (scale) and through different
simulated "instruments" (contrast, noise, blur), so that a model must learn
representations that are robust to scale and modality shift.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

H0, K_HP = 1.0, 2.0  # hardness proxy constants (arbitrary units)

INSTRUMENTS = {
    # name: (boundary contrast, gaussian noise std, blur sigma, gamma)
    "A": (1.0, 0.05, 0.6, 1.0),   # clean, sharp (source instrument)
    "B": (0.6, 0.15, 1.4, 0.7),   # noisier, blurrier, lower contrast (target)
}


def render(grain_um, scale_um_per_px, instrument, rng, size=64):
    """Render one micrograph of a microstructure with mean grain size grain_um."""
    field_um = size * scale_um_per_px
    n_grains = max(4, int((field_um / grain_um) ** 2))
    seeds = rng.uniform(0, size, size=(n_grains, 2))
    yy, xx = np.mgrid[0:size, 0:size]
    pts = np.stack([yy.ravel(), xx.ravel()], 1)
    _, label = cKDTree(seeds).query(pts)
    label = label.reshape(size, size)
    shade = rng.uniform(0.3, 0.8, size=n_grains)        # per-grain orientation contrast
    img = shade[label]
    boundary = (np.abs(np.diff(label, axis=0, prepend=label[:1])) > 0) | \
               (np.abs(np.diff(label, axis=1, prepend=label[:, :1])) > 0)
    contrast, noise, blur, gamma = INSTRUMENTS[instrument]
    img = img - contrast * 0.5 * boundary
    img = ndimage.gaussian_filter(img, blur)
    img = np.clip(img, 0, 1) ** gamma
    img = img + rng.normal(0, noise, img.shape)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img[::2, ::2].astype(np.float32)  # 32x32


def make_dataset(n, instrument, seed=0):
    rng = np.random.default_rng(seed)
    grain = rng.uniform(1.0, 8.0, n)                  # mean grain size, micrometres
    scale = rng.choice([0.25, 0.5, 1.0], n)           # magnification (um / px)
    X = np.stack([render(g, s, instrument, rng) for g, s in zip(grain, scale)])
    y = H0 + K_HP / np.sqrt(grain)
    return X, y, scale
