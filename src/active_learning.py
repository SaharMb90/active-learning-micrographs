"""Pool-based active learning with a Gaussian-process surrogate."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler


def fit_gp(Z, y):
    k = C(1.0) * RBF(np.sqrt(Z.shape[1])) + WhiteKernel(1e-2, (1e-4, 1.0))
    return GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(Z, y)


def run(Z_pool, y_pool, Z_test, y_test, strategy, budget=60, init=8, step=4, seed=0):
    """Returns list of (n_labels, rmse, coverage_90) on the test set."""
    rng = np.random.default_rng(seed)
    sc = StandardScaler().fit(Z_pool)
    Zp, Zt = sc.transform(Z_pool), sc.transform(Z_test)
    labelled = list(rng.choice(len(Zp), init, replace=False))
    out = []
    while True:
        gp = fit_gp(Zp[labelled], y_pool[labelled])
        mu, sd = gp.predict(Zt, return_std=True)
        rmse = float(np.sqrt(np.mean((mu - y_test) ** 2)))
        cover = float(np.mean(np.abs(mu - y_test) <= 1.645 * sd))
        out.append((len(labelled), rmse, cover))
        if len(labelled) >= budget:
            return out
        pool = np.setdiff1d(np.arange(len(Zp)), labelled)
        if strategy == "uncertainty":
            _, s = gp.predict(Zp[pool], return_std=True)
            pick = pool[np.argsort(-s)[:step]]
        else:
            pick = rng.choice(pool, step, replace=False)
        labelled += list(pick)
