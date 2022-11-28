import numpy as np
from scipy.stats import multivariate_normal
import torch


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)
    z = z / z.max()
    z = z.astype(np.float32)
    #mask = torch.from_numpy(z)

    return z


class ImgsTracker:

    def __init__(self) -> None:
        pass
