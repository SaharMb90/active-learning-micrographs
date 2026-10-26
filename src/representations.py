"""Image representations compared in the demonstrator.

Every representation gets the known magnification (log um/px) appended,
because real microscopes record it in the image metadata.
"""
import warnings
import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from skimage.feature import local_binary_pattern

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def multiscale_texture(img):
    """Multi-scale texture statistics: gradient energy + LBP histogram per scale."""
    feats = []
    for sigma in (0.5, 1.0, 2.0):
        sm = ndimage.gaussian_filter(img, sigma)
        g = np.hypot(ndimage.sobel(sm, 0), ndimage.sobel(sm, 1))
        feats += [g.mean(), g.std(), np.percentile(g, 90)]
        u8 = ((sm - sm.min()) / (np.ptp(sm) + 1e-8) * 255).astype(np.uint8)
        lbp = local_binary_pattern(u8, 8, 1, "uniform")
        feats += list(np.histogram(lbp, bins=10, range=(0, 10), density=True)[0])
    return np.array(feats)


class PixelPCA:
    """Baseline: 16-component PCA on raw pixels (fit on source images)."""
    name = "pixel_pca"

    def fit(self, X_src, scale_src, X_tgt=None, scale_tgt=None):
        self.pca = PCA(16, random_state=0).fit(X_src.reshape(len(X_src), -1))
        Z = self.pca.transform(X_src.reshape(len(X_src), -1))
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-8
        return self

    def transform(self, X, scale, domain="source"):
        Z = (self.pca.transform(X.reshape(len(X), -1)) - self.mu) / self.sd
        return np.c_[Z, np.log(scale)]


class Texture:
    """Multi-scale texture features standardised with SOURCE statistics only."""
    name = "texture"

    def fit(self, X_src, scale_src, X_tgt=None, scale_tgt=None):
        H = np.stack([multiscale_texture(x) for x in X_src])
        self.stats = {"source": (H.mean(0), H.std(0) + 1e-8)}
        return self

    def transform(self, X, scale, domain="source"):
        mu, sd = self.stats["source"]
        H = np.stack([multiscale_texture(x) for x in X])
        return np.c_[(H - mu) / sd, np.log(scale)]


class TextureAligned(Texture):
    """Same features, but each instrument is standardised with its OWN statistics,
    estimated from unlabelled images (no target labels used). This is a simple
    form of unsupervised domain alignment."""
    name = "texture_aligned"

    def fit(self, X_src, scale_src, X_tgt, scale_tgt):
        super().fit(X_src, scale_src)
        H = np.stack([multiscale_texture(x) for x in X_tgt])
        self.stats["target"] = (H.mean(0), H.std(0) + 1e-8)
        return self

    def transform(self, X, scale, domain="source"):
        mu, sd = self.stats[domain]
        H = np.stack([multiscale_texture(x) for x in X])
        return np.c_[(H - mu) / sd, np.log(scale)]
