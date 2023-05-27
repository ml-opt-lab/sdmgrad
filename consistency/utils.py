import numpy as np
from min_norm_solvers import MinNormSolver
from scipy.optimize import minimize, Bounds, minimize_scalar

import torch
from torch import linalg as LA
from torch.nn import functional as F


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    [2] Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application
        Weiran Wang, Miguel Á. Carreira-Perpiñán. arXiv:1309.1541
        https://arxiv.org/pdf/1309.1541.pdf
    [3] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246#file-simplex_projection-py
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = v.astype(np.float64)
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def grad2vec(m, grads, grad_dims, task):
    # store the gradients
    grads[:, task].fill_(0.0)
    cnt = 0
    for mm in m.shared_modules():
        for p in mm.parameters():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg:en, task].copy_(grad_cur.data.view(-1))
            cnt += 1


def overwrite_grad(m, newgrad, grad_dims):
    # newgrad = newgrad * 2 # to match the sum loss
    cnt = 0
    for mm in m.shared_modules():
        for param in mm.parameters():
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg:en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1


def mean_grad(grads):
    return grads.mean(1)


def mgd(grads):
    grads_cpu = grads.t().cpu()
    sol, min_norm = MinNormSolver.find_min_norm_element([grads_cpu[t] for t in range(grads.shape[-1])])
    w = torch.FloatTensor(sol).to(grads.device)
    g = grads.mm(w.view(-1, 1)).view(-1)
    return g


def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:, 0]
    g2 = grads[:, 1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        return coef * np.sqrt(x**2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 +
                              1e-8) + 0.5 * x * (g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22

    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x**2 * g11 + (1 - x)**2 * g22 + 2 * x * (1 - x) * g12 + 1e-8)
    lmbda = coef / (gw_norm + 1e-8)
    g = (0.5 + lmbda * x) * g1 + (0.5 + lmbda * (1 - x)) * g2  # g0 + lmbda*gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha**2)
    else:
        return g / (1 + alpha)


def sdmgrad(w, grads, lmbda, niter=20):
    """
    our proposed sdmgrad
    """
    GG = torch.mm(grads.t(), grads)
    scale = torch.mean(torch.sqrt(torch.diag(GG) + 1e-4))
    GG = GG / scale.pow(2)
    Gg = torch.mean(GG, dim=1)
    gg = torch.mean(Gg)

    w.requires_grad = True
    optimizer = torch.optim.SGD([w], lr=10, momentum=0.5)
    for i in range(niter):
        optimizer.zero_grad()
        obj = torch.dot(w, torch.mv(GG, w)) + 2 * lmbda * torch.dot(w, Gg) + lmbda**2 * gg
        obj.backward()
        optimizer.step()
        proj = euclidean_proj_simplex(w.data.cpu().numpy())
        w.data.copy_(torch.from_numpy(proj).data)
    w.requires_grad = False

    g0 = torch.mean(grads, dim=1)
    gw = torch.mv(grads, w)
    g = (gw + lmbda * g0) / (1 + lmbda)
