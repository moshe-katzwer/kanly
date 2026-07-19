import pandas as pd
import numpy as np

# #### #
# NLLS #
def get_nlls_data():
    np.random.seed(0)
    n = 500
    nlls_data = pd.DataFrame({'x': np.random.randn(n)*.25})
    nlls_data['y'] = np.exp(-1.1 + 2.2 * nlls_data.x + .6 * np.random.randn(n))
    return nlls_data

# ## #
# QR #
def get_qr_data():
    np.random.seed(0)
    n = 3500
    qr_data = pd.DataFrame()
    qr_data['x'] = np.random.randn(n)
    qr_data['y'] = np.exp(-.6 + .35 * qr_data.x + .4 * np.random.randn(n))
    return qr_data

# ### #
# GMM #
# ### #
def get_gmm_data():
    np.random.seed(0)
    n = 500
    gmm_data = pd.DataFrame({'z': np.random.randn(n)})
    gmm_data['e'] = np.random.randn(n)
    gmm_data['x'] = np.exp(-.13 + gmm_data.z - .2 * gmm_data.z**2 + .5 * np.random.randn(n) + .75 * gmm_data.e)
    gmm_data['y'] = -1.5 + 2.2 * gmm_data.x + gmm_data.e
    return gmm_data


# ########### #
# ElASTIC NET #
# ########### #
def get_elastic_net_data():
    np.random.seed(0)
    n = 500
    p = 500
    X = np.random.randn(n, p).dot(np.random.randn(p, p))
    data = pd.DataFrame(X, columns=[f'x{j}' for j in range(p)])
    beta = np.zeros(p)
    beta[[5, 20, 30, 104]] = np.arange(1, 5)
    data['y'] = 3 * np.random.randn(n) + 10.5 + X.dot(beta)
    return data
