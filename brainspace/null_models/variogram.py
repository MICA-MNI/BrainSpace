"""
Implementation of variogram-matching procedure.
"""

# Author: Joshua Burt <joshua.burt@yale.edu>
# License: BSD 3 clause

import numpy as np
import numpy.lib.format
from pathlib import Path
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


# ----------------------
# ------ Checks --------
# ----------------------

def is_string_like(obj):
    """ Check whether `obj` behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


def check_map(x):
    """
    Check that brain map is array_like and one dimensional.

    Parameters
    ----------
    x : 1D ndarray
        Brain map

    Returns
    -------
    None

    Raises
    ------
    TypeError : `x` is not a ndarray object
    ValueError : `x` is not one-dimensional

    """
    if not isinstance(x, np.ndarray):
        e = "Brain map must be array-like\n"
        e += "got type {}".format(type(x))
        raise TypeError(e)
    if x.ndim != 1:
        e = "Brain map must be one-dimensional\n"
        e += "got shape {}".format(x.shape)
        raise ValueError(e)


def check_pv(pv):
    """
    Check input argument `pv`.

    Parameters
    ----------
    pv : int
        Percentile of the pairwise distance distribution at which to truncate
        during variogram fitting.

    Returns
    -------
    int

    Raises
    ------
    ValueError : `pv` lies outside range (0, 100]

    """
    try:
        pv = int(pv)
    except ValueError:
        raise ValueError("parameter 'pv' must be an integer in (0,100]")
    if pv <= 0 or pv > 100:
        raise ValueError("parameter 'pv' must be in (0,100]")
    return pv


def check_deltas(deltas):
    """
    Check input argument `deltas`.

    Parameters
    ----------
    deltas : 1D ndarray or List[float]
        Proportions of neighbors to include for smoothing, in (0, 1]

    Returns
    -------
    None

    Raises
    ------
    TypeError : `deltas` is not a List or ndarray object
    ValueError : One or more elements of `deltas` lies outside (0,1]

    """
    if not isinstance(deltas, list) and not isinstance(deltas, np.ndarray):
        raise TypeError("Parameter `deltas` must be a list or ndarray")
    for d in deltas:
        if d <= 0 or d > 1:
            raise ValueError("Each element of `deltas` must lie in (0,1]")


def count_lines(filename):
    """
    Count number of lines in a file.

    Parameters
    ----------
    filename : filename

    Returns
    -------
    int
        number of lines in file

    """
    with open(filename, 'rb') as f:
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
        return lines

# ----------------------
# ------ Data I/O ------
# ----------------------


def dataio(x):
    """
    Data I/O for core classes.

    To facilitate flexible user inputs, this function loads data from:
        - txt files
        - npy files (memory-mapped arrays)
        - array_like data

    Parameters
    ----------
    x : filename or ndarray or np.memmap

    Returns
    -------
    ndarray or np.memmap

    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined by file extension
    TypeError : input is not a filename or array_like object

    """
    if is_string_like(x):
        if not Path(x).exists():
            raise FileExistsError("file does not exist: {}".format(x))
        elif Path(x).stat().st_size == 0:
            raise RuntimeError("file is empty: {}".format(x))
        elif Path(x).suffix == ".npy":  # memmap
            return np.load(x, mmap_mode='r')
        elif Path(x).suffix == ".txt":  # text file
            return np.loadtxt(x).squeeze()
        else:
            raise ValueError(
                "expected npy or txt file, got {}".format(Path(x).suffix))
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "expected filename or array_like obj, got {}".format(type(x)))
        return x


def txt2memmap(dist_file, output_dir, maskfile=None, delimiter=' '):
    """
    Export distance matrix to memory-mapped array.

    Parameters
    ----------
    dist_file : filename
        Path to `delimiter`-separated distance matrix file
    output_dir : filename
        Path to directory in which output files will be written
    maskfile : filename or ndarray or None, default None
        Path to a neuroimaging/txt file containing a mask, or a mask
        represented as a numpy array. Mask scalars are cast to boolean, and
        all elements not equal to zero will be masked.
    delimiter : str
        Delimiting character in `dist_file`

    Returns
    -------
    dict
        Keys are 'D' and 'index'; values are absolute paths to the
        corresponding binary files on disk.

    Notes
    -----
    Each row of the distance matrix is sorted before writing to file. Thus, a
    second mem-mapped array is necessary, the i-th row of which contains
    argsort(d[i]).
    If `maskfile` is not None, a binary mask.txt file will also be written to
    the output directory.

    Raises
    ------
    IOError : `output_dir` doesn't exist
    ValueError : Mask image and distance matrix have inconsistent sizes

    """
    op = Path(output_dir)
    nlines = count_lines(dist_file)
    if not op.exists():
        raise IOError("Output directory does not exist: {}".format(output_dir))

    # Load mask if one was provided
    if maskfile is not None:
        mask = dataio(maskfile).astype(bool)
        if mask.size != nlines:
            e = "Incompatible input sizes\n"
            e += "{} rows in {}\n".format(nlines, dist_file)
            e += "{} elements in {}".format(mask.size, maskfile)
            raise ValueError(e)
        mask_fileout = str(op.joinpath("mask.txt"))
        np.savetxt(  # Write to text file
            fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
        nv = int((~mask).sum())  # number of non-masked elements
        idx = np.arange(nlines)[~mask]  # indices of non-masked elements
    else:
        nv = nlines
        idx = np.arange(nlines)

    # Build memory-mapped arrays
    with open(dist_file, 'r') as fp:

        npydfile = str(op.joinpath("distmat.npy"))
        npyifile = str(op.joinpath("index.npy"))
        fpd = numpy.lib.format.open_memmap(
            npydfile, mode='w+', dtype=np.float32, shape=(nv, nv))
        fpi = numpy.lib.format.open_memmap(
            npyifile, mode='w+', dtype=np.int32, shape=(nv, nv))

        ifp = 0  # Build memory-mapped arrays one row of distances at a time
        for il, l in enumerate(fp):  # Loop over lines of file
            if il not in idx:  # Keep only CIFTI vertices
                continue
            else:
                line = l.rstrip()
                if line:
                    data = np.array(line.split(delimiter), dtype=np.float32)
                    if data.size != nlines:
                        raise RuntimeError(
                            "Distance matrix is not square: {}".format(
                                dist_file))
                    d = data[idx]
                    sort_idx = np.argsort(d)
                    fpd[ifp, :] = d[sort_idx]  # sorted row of distances
                    fpi[ifp, :] = sort_idx  # sort indexes
                    ifp += 1
        del fpd  # Flush memory changes to disk
        del fpi

    return {'distmat': npydfile, 'index': npyifile}  # Return filenames

# ----------------------
# - Smoothing kernels --
# ----------------------


def gaussian(d):
    """
    Gaussian kernel which truncates at one standard deviation.

    Parameters
    ----------
    d : ndarray, shape (N,) or (M,N)
        one- or two-dimensional array of distances

    Returns
    -------
    ndarray, shape (N,) or (M,N)
        Gaussian kernel weights

    Raises
    ------
    TypeError : `d` is not array_like

    """
    try:  # 2-dim
        return np.exp(-1.25 * np.square(d / d.max(axis=-1)[:, np.newaxis]))
    except IndexError:  # 1-dim
        return np.exp(-1.25 * np.square(d/d.max()))
    except AttributeError:
        raise TypeError("expected array_like, got {}".format(type(d)))


def exp(d):
    """
    Exponentially decaying kernel which truncates at e^{-1}.

    Parameters
    ----------
    d : ndarray, shape (N,) or (M,N)
        one- or two-dimensional array of distances

    Returns
    -------
    ndarray, shape (N,) or (M,N)
        Exponential kernel weights

    Notes
    -----
    Characteristic length scale is set to d.max(axis=-1), i.e. the maximum
    distance within each row.

    Raises
    ------
    TypeError : `d` is not array_like

    """
    try:  # 2-dim
        return np.exp(-d / d.max(axis=-1)[:, np.newaxis])
    except IndexError:  # 1-dim
        return np.exp(-d/d.max())
    except AttributeError:
        raise TypeError("expected array_like, got {}".format(type(d)))


def invdist(d):
    """
    Inverse distance kernel.

    Parameters
    ----------
    d : ndarray, shape (N,) or (M,N)
        One- or two-dimensional array of distances

    Returns
    -------
    ndarray, shape (N,) or (M,N)
        Inverse distance, i.e. d^{-1}

    Raises
    ------
    ZeroDivisionError : `d` includes zero value
    TypeError : `d` is not array_like

    """
    try:
        return 1. / d
    except ZeroDivisionError as e:
        raise ZeroDivisionError(e)
    except AttributeError:
        raise TypeError("expected array_like, got {}".format(type(d)))


def uniform(d):
    """
    Uniform (i.e., distance independent) kernel.

    Parameters
    ----------
    d : ndarray, shape (N,) or (M,N)
        One- or two-dimensional array of distances

    Returns
    -------
    ndarray, shape (N,) or (M,N)
        Uniform kernel weights

    Notes
    -----
    Each element is normalized to 1/N such that columns sum to unity.

    Raises
    ------
    TypeError : `d` is not array_like

    """
    try:  # 2-dim
        return np.ones(d.shape) / d.shape[-1]
    except IndexError:  # 1-dim
        return np.ones(d.size) / d.size
    except AttributeError:
        raise TypeError("expected array_like, got {}".format(type(d)))


def check_kernel(kernel):
    """
    Check that a valid kernel was specified and return callable.

    Parameters
    ----------
    kernel : 'exp' or 'gaussian' or 'invdist' or 'uniform'
        Kernel selection

    Returns
    -------
    Callable

    Raises
    ------
    NotImplementedError : kernel is not implemented

    """
    kernels = {'exp': exp,
               'gaussian': gaussian,
               'invdist': invdist,
               'uniform': uniform}
    if kernel not in kernels.keys():
        e = "'{}' is not a valid kernel\n".format(kernel)
        e += "Valid kernels: {}".format(", ".join([k for k in kernels.keys()]))
        raise NotImplementedError(e)
    return kernels[kernel]

# ----------------------
# ---- Core classes ----
# ----------------------


# class Base:
#     """ Base implementation of map generator.
#
#     Parameters
#     ----------
#     x : filename or 1D ndarray
#         Target brain map
#     D : filename or ndarray, shape (N,N)
#         Pairwise distance matrix
#     deltas : 1D ndarray or List[float], default [0.1,0.2,...,0.9]
#         Proportion of neighbors to include for smoothing, in (0, 1]
#     kernel : str, default 'exp'
#         Kernel with which to smooth permuted maps:
#           'gaussian' : Gaussian function.
#           'exp' : Exponential decay function.
#           'invdist' : Inverse distance.
#           'uniform' : Uniform weights (distance independent).
#     pv : int, default 25
#         Percentile of the pairwise distance distribution at which to
#         truncate during variogram fitting
#     nh : int, default 25
#         Number of uniformly spaced distances at which to compute variogram
#     resample : bool, default False
#         Resample surrogate maps' values from target brain map
#     b : float or None, default None
#         Gaussian kernel bandwidth for variogram smoothing. If None, set to
#         three times the spacing between variogram x-coordinates.
#
#     Notes
#     -----
#     Passing resample=True preserves the distribution of values in the target
#     map, with the possibility of worsening the simulated surrogate maps'
#     variograms fits.
#
#     """
#
#     def __init__(self, x, D, deltas=np.linspace(0.1, 0.9, 9),
#                  kernel='exp', pv=25, nh=25, resample=False, b=None):
#
#         self.x = x
#         self.D = D
#         n = self._x.size
#         self.resample = resample
#         self.nh = nh
#         self.deltas = deltas
#         self.pv = pv
#         self.nmap = n
#         self.kernel = kernel  # Smoothing kernel selection
#         self._ikn = np.arange(n)[:, None]
#         self._triu = np.triu_indices(self._nmap, k=1)  # upper triangular inds
#         self._u = self._D[self._triu]  # variogram X-coordinate
#         self._v = self.compute_variogram(self._x)  # variogram Y-coord
#
#         # Get indices of pairs with u < pv'th percentile
#         self._uidx = np.where(self._u < np.percentile(self._u, self._pv))[0]
#         self._uisort = np.argsort(self._u[self._uidx])
#
#         # Find sorted indices of first `kmax` elements of each row of dist. mat.
#         self._disort = np.argsort(self._D, axis=-1)
#         self._jkn = dict.fromkeys(deltas)
#         self._dkn = dict.fromkeys(deltas)
#         for delta in deltas:
#             k = int(delta*n)
#             # find index of k nearest neighbors for each area
#             self._jkn[delta] = self._disort[:, 1:k+1]  # prevent self-coupling
#             # find distance to k nearest neighbors for each area
#             self._dkn[delta] = self._D[(self._ikn, self._jkn[delta])]
#
#         # Smoothed variogram and variogram _b
#         utrunc = self._u[self._uidx]
#         self._h = np.linspace(utrunc.min(), utrunc.max(), self._nh)
#         self.b = b
#         self._smvar = self.smooth_variogram(self._v)
#
#         # Linear regression model
#         self._lm = LinearRegression(fit_intercept=True)
#
#     def __call__(self, n=1):
#         """
#         Randomly generate new surrogate map(s).
#
#         Parameters
#         ----------
#         n : int, default 1
#             Number of surrogate maps to randomly generate
#
#         Returns
#         -------
#         ndarray, shape (n,N)
#             Randomly generated map(s) with matched spatial autocorrelation
#
#         Notes
#         -----
#         Chooses a level of smoothing that produces a smoothed variogram which
#         best approximates the true smoothed variogram. Selecting resample='True'
#         preserves the original map's value distribution at the expense of
#         worsening the surrogate maps' variogram fit.
#
#         """
#         print("Generating {} maps...".format(n))
#         surrs = np.empty((n, self._nmap))
#         for i in range(n):  # generate random maps
#
#             xperm = self.permute_map()  # Randomly permute values
#             res = dict.fromkeys(self._deltas)
#
#             for delta in self.deltas:  # foreach neighborhood size
#                 # Smooth the permuted map using delta proportion of
#                 # neighbors to reintroduce spatial autocorrelation
#                 sm_xperm = self.smooth_map(x=xperm, delta=delta)
#
#                 # Calculate empirical variogram of the smoothed permuted map
#                 vperm = self.compute_variogram(sm_xperm)
#
#                 # Calculate smoothed variogram of the smoothed permuted map
#                 smvar_perm = self.smooth_variogram(vperm)
#
#                 # Fit linear regression btwn smoothed variograms
#                 res[delta] = self.regress(smvar_perm, self._smvar)
#
#             alphas, betas, residuals = np.array(
#                 [res[d] for d in self._deltas]).T
#
#             # Select best-fit model and regression parameters
#             iopt = np.argmin(residuals)
#             dopt = self._deltas[iopt]
#             aopt = alphas[iopt]
#             bopt = betas[iopt]
#
#             # Transform and smooth permuted map using best-fit parameters
#             sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
#             surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
#                     np.sqrt(np.abs(aopt)) * np.random.randn(self._nmap))
#             surrs[i] = surr
#
#         if self._resample:  # resample values from empirical map
#             sorted_map = np.sort(self._x)
#             for i, surr in enumerate(surrs):
#                 ii = np.argsort(surr)
#                 np.put(surr, ii, sorted_map)
#
#         return surrs.squeeze()
#
#     def compute_variogram(self, x):
#         """
#         Compute variogram values (i.e., one-half squared pairwise differences).
#
#         Parameters
#         ----------
#         x : 1D ndarray
#             Brain map scalar array
#
#         Returns
#         -------
#         v : ndarray, shape (N(N-1)/2,)
#            Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2
#
#         """
#         diff_ij = np.subtract.outer(x, x)
#         v = 0.5 * np.square(diff_ij)[self._triu]
#         return v
#
#     def permute_map(self):
#         """
#         Return randomly permuted brain map.
#
#         Returns
#         -------
#         1D ndarray
#             Random permutation of target brain map
#
#         """
#         perm_idx = np.random.permutation(np.arange(self._x.size))
#         mask_perm = self._x.mask[perm_idx]
#         x_perm = self._x.data[perm_idx]
#         return np.ma.masked_array(data=x_perm, mask=mask_perm)
#
#     def smooth_map(self, x, delta):
#         """
#         Smooth `x` using `delta` proportion of nearest neighbors.
#
#         Parameters
#         ----------
#         x : 1D ndarray
#             Brain map scalars
#         delta : float
#             Proportion of neighbors to include for smoothing, in (0, 1)
#
#         Returns
#         -------
#         1D ndarray
#             Smoothed brain map
#
#         """
#         # Values of k nearest neighbors for each brain area
#         xkn = x[self._jkn[delta]]
#         weights = self._kernel(self._dkn[delta])  # Distance-weight kernel
#         # Kernel-weighted sum
#         return (weights * xkn).sum(axis=1) / weights.sum(axis=1)
#
#     def smooth_variogram(self, v, return_h=False):
#         """
#         Smooth a variogram.
#
#         Parameters
#         ----------
#         v : 1D ndarray
#             Variogram values, i.e. 0.5 * (x_i - x_j) ^ 2
#         return_h : bool, default False
#             Return distances at which the smoothed variogram was computed
#
#         Returns
#         -------
#         1D ndarray, shape (nh,)
#             Smoothed variogram values
#         1D ndarray, shape (nh,)
#             Distances at which smoothed variogram was computed (returned only if
#             `return_h` is True)
#
#         Raises
#         ------
#         ValueError : `v` has unexpected size.
#
#         """
#         u = self._u[self._uidx]
#         v = v[self._uidx]
#         if len(u) != len(v):
#             raise ValueError(
#                 "argument v: expected size {}, got {}".format(len(u), len(v)))
#         # Subtract each h from each pairwise distance u
#         # Each row corresponds to a unique h
#         du = np.abs(u - self._h[:, None])
#         w = np.exp(-np.square(2.68 * du / self._b) / 2)
#         denom = w.sum(axis=1)
#         wv = w * v[None, :]
#         num = wv.sum(axis=1)
#         output = num / denom
#         if not return_h:
#             return output
#         return output, self._h
#
#     def regress(self, x, y):
#         """
#         Linearly regress `x` onto `y`.
#
#         Parameters
#         ----------
#         x : 1D ndarray
#             Independent variable
#         y : 1D ndarray
#             Dependent variable
#
#         Returns
#         -------
#         alpha : float
#             Intercept term (offset parameter)
#         beta : float
#             Regression coefficient (scale parameter)
#         res : float
#             Sum of squared residuals
#
#         """
#         self._lm.fit(X=np.expand_dims(x, -1), y=y)
#         beta = self._lm.coef_
#         alpha = self._lm.intercept_
#         y_pred = self._lm.predict(X=np.expand_dims(x, -1))
#         res = np.sum(np.square(y-y_pred))
#         return alpha, beta, res
#
#     @property
#     def x(self):
#         """ 1D ndarray : brain map scalar array """
#         return self._x
#
#     @x.setter
#     def x(self, x):
#         x_ = dataio(x)
#         check_map(x=x_)
#         brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
#         self._x = brain_map
#
#     @property
#     def D(self):
#         """ ndarray, shape (N,N) : Pairwise distance matrix """
#         return self._D
#
#     @D.setter
#     def D(self, x):
#         d_ = dataio(x)
#         if not np.allclose(d_, d_.T):
#             raise ValueError("Distance matrix must be symmetric")
#         n = self._x.size
#         if d_.shape != (n, n):
#             e = "Distance matrix must have dimensions consistent with brain map"
#             e += "\nDistance matrix shape: {}".format(d_.shape)
#             e += "\nBrain map size: {}".format(n)
#             raise ValueError(e)
#         self._D = d_
#
#     @property
#     def nmap(self):
#         """ int : length of brain map """
#         return self._nmap
#
#     @nmap.setter
#     def nmap(self, x):
#         self._nmap = int(x)
#
#     @property
#     def pv(self):
#         """ int : percentile of pairwise distances at which to truncate """
#         return self._pv
#
#     @pv.setter
#     def pv(self, x):
#         pv = check_pv(x)
#         self._pv = pv
#
#     @property
#     def deltas(self):
#         """ 1D ndarray or List[float] : proportions of nearest neighbors """
#         return self._deltas
#
#     @deltas.setter
#     def deltas(self, x):
#         check_deltas(deltas=x)
#         self._deltas = x
#
#     @property
#     def nh(self):
#         """ int : number of variogram distance intervals """
#         return self._nh
#
#     @nh.setter
#     def nh(self, x):
#         self._nh = x
#
#     @property
#     def h(self):
#         """ 1D ndarray : distances at which smoothed variogram is computed """
#         return self._h
#
#     @property
#     def kernel(self):
#         """ Callable : smoothing kernel function """
#         return self._kernel
#
#     @kernel.setter
#     def kernel(self, x):
#         kernel_callable = check_kernel(x)
#         self._kernel = kernel_callable
#
#     @property
#     def resample(self):
#         """ bool : whether to resample surrogate maps from target map """
#         return self._resample
#
#     @resample.setter
#     def resample(self, x):
#         if not isinstance(x, bool):
#             e = "parameter `resample`: expected bool, got {}".format(type(x))
#             raise TypeError(e)
#         self._resample = x
#
#     @property
#     def b(self):
#         """ numeric : Gaussian kernel bandwidth """
#         return self._b
#
#     @b.setter
#     def b(self, x):
#         if x is not None:
#             try:
#                 self._b = float(x)
#             except (ValueError, TypeError):
#                 e = "bandwidth b: expected numeric, got {}".format(type(x))
#                 raise ValueError(e)
#         else:   # set bandwidth equal to 3x bin spacing
#             self._b = 3.*np.mean(self._h[1:] - self._h[:-1])
#
#
# class Sampled:
#     """
#     Sampling implementation of map generator.
#
#     Parameters
#     ----------
#     x : 1D ndarray
#         Target brain map
#     D : ndarray or memmap, shape (N,N)
#         Pairwise distance matrix between elements of `x`. Each row of `D` should
#         be sorted. Indices used to sort each row are passed to the `index`
#         argument. See :func:`brainsmash.mapgen.memmap.txt2memmap` or the online
#         documentation for more details (brainsmash.readthedocs.io)
#     index : filename or ndarray or memmap, shape(N,N)
#         See above
#     ns : int, default 500
#         Take a subsample of `ns` rows from `D` when fitting variograms
#     deltas : ndarray or List[float], default [0.3, 0.5, 0.7, 0.9]
#         Proportions of neighbors to include for smoothing, in (0, 1]
#     kernel : str, default 'exp'
#         Kernel with which to smooth permuted maps
#         - 'gaussian' : gaussian function
#         - 'exp' : exponential decay function
#         - 'invdist' : inverse distance
#         - 'uniform' : uniform weights (distance independent)
#     pv : int, default 70
#         Percentile of the pairwise distance distribution (in `D`) at
#         which to truncate during variogram fitting
#     nh : int, default 25
#         Number of uniformly spaced distances at which to compute variogram
#     knn : int, default 1000
#         Number of nearest regions to keep in the neighborhood of each region
#     b : float or None, default None
#         Gaussian kernel bandwidth for variogram smoothing. if None,
#         three times the distance interval spacing is used.
#     resample : bool, default False
#         Resample surrogate map values from the target brain map
#     verbose : bool, default False
#         Print surrogate count each time new surrogate map created
#
#     Notes
#     -----
#     Passing resample=True will preserve the distribution of values in the
#     target map, at the expense of worsening simulated surrogate maps'
#     variograms fits. This worsening will increase as the empirical map
#     more strongly deviates from normality.
#
#     Raises
#     ------
#     ValueError : `x` and `D` have inconsistent sizes
#
#     """
#
#     def __init__(self, x, D, index, ns=500, pv=70, nh=25, knn=1000, b=None,
#                  deltas=np.arange(0.3, 1., 0.2), kernel='exp', resample=False,
#                  verbose=False):
#
#         self._verbose = verbose
#         self.x = x
#         n = self._x.size
#         self.nmap = int(n)
#         self.knn = knn
#         self.D = D
#         self.index = index
#         self.resample = resample
#         self.nh = int(nh)
#         self.deltas = deltas
#         self.ns = int(ns)
#         self.b = b
#         self.pv = pv
#         self._ikn = np.arange(self._nmap)[:, None]
#
#         # Store k nearest neighbors from distance and index matrices
#         self.kernel = kernel  # Smoothing kernel selection
#         self._dmax = np.percentile(self._D, self._pv)
#         self.h = np.linspace(self._D.min(), self._dmax, self._nh)
#         if not self._b:
#             self.b = 3 * (self.h[1] - self.h[0])
#
#         # Linear regression model
#         self._lm = LinearRegression(fit_intercept=True)
#
#     def __call__(self, n=1):
#         """
#         Randomly generate new surrogate map(s).
#
#         Parameters
#         ----------
#         n : int, default 1
#             Number of surrogate maps to randomly generate
#
#         Returns
#         -------
#         ndarray, shape (n,N)
#             Randomly generated map(s) with matched spatial autocorrelation
#
#         Notes
#         -----
#         Chooses a level of smoothing that produces a smoothed variogram which
#         best approximates the true smoothed variogram. Selecting resample='True'
#         preserves the map value distribution at the expense of worsening the
#         surrogate maps' variogram fits.
#
#         """
#         if self._verbose:
#             print("Generating {} maps...".format(n))
#         surrs = np.empty((n, self._nmap))
#         for i in range(n):  # generate random maps
#             if self._verbose:
#                 print(i+1)
#
#             # Randomly permute map
#             x_perm = self.permute_map()
#
#             # Randomly select subset of regions to use for variograms
#             idx = self.sample()
#
#             # Compute empirical variogram
#             v = self.compute_variogram(self._x, idx)
#
#             # Variogram ordinates; use nearest neighbors because local effect
#             u = self._D[idx, :]
#             uidx = np.where(u < self._dmax)
#
#             # Smooth empirical variogram
#             smvar, u0 = self.smooth_variogram(u[uidx], v[uidx], return_h=True)
#
#             res = dict.fromkeys(self._deltas)
#
#             for d in self._deltas:  # foreach neighborhood size
#
#                 k = int(d * self._knn)
#
#                 # Smooth the permuted map using k nearest neighbors to
#                 # reintroduce spatial autocorrelation
#                 sm_xperm = self.smooth_map(x=x_perm, k=k)
#
#                 # Calculate variogram values for the smoothed permuted map
#                 vperm = self.compute_variogram(sm_xperm, idx)
#
#                 # Calculate smoothed variogram of the smoothed permuted map
#                 smvar_perm = self.smooth_variogram(u[uidx], vperm[uidx])
#
#                 # Fit linear regression btwn smoothed variograms
#                 res[d] = self.regress(smvar_perm, smvar)
#
#             alphas, betas, residuals = np.array(
#                 [res[d] for d in self._deltas]).T
#
#             # Select best-fit model and regression parameters
#             iopt = np.argmin(residuals)
#             dopt = self._deltas[iopt]
#             self._dopt = dopt
#             kopt = int(dopt * self._knn)
#             aopt = alphas[iopt]
#             bopt = betas[iopt]
#
#             # Transform and smooth permuted map using best-fit parameters
#             sm_xperm_best = self.smooth_map(x=x_perm, k=kopt)
#             surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
#                     np.sqrt(np.abs(aopt)) * np.random.randn(self._nmap))
#             surrs[i] = surr
#
#         if self._resample:  # resample values from empirical map
#             sorted_map = np.sort(self._x)
#             for i, surr in enumerate(surrs):
#                 ii = np.argsort(surr)
#                 np.put(surr, ii, sorted_map)
#
#         if self._ismasked:
#             return np.ma.masked_array(
#                 data=surrs, mask=np.isnan(surrs)).squeeze()
#         return surrs.squeeze()
#
#     def compute_variogram(self, x, idx):
#         """
#         Compute variogram of `x` using pairs of regions indexed by `idx`.
#
#         Parameters
#         ----------
#         x : 1Dndarray
#             Brain map
#         idx : ndarray[int], shape (ns,)
#             Indices of randomly sampled brain regions
#
#         Returns
#         -------
#         v : ndarray, shape (ns,ns)
#             Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2, for i,j in idx
#
#         """
#         diff_ij = x[idx][:, None] - x[self._index[idx, :]]
#         return 0.5 * np.square(diff_ij)
#
#     def permute_map(self):
#         """
#         Return a random permutation of the target brain map.
#
#         Returns
#         -------
#         1D ndarray
#             Random permutation of target brain map
#
#         """
#         perm_idx = np.random.permutation(self._nmap)
#         if self._ismasked:
#             mask_perm = self._x.mask[perm_idx]
#             x_perm = self._x.data[perm_idx]
#             return np.ma.masked_array(data=x_perm, mask=mask_perm)
#         return self._x[perm_idx]
#
#     def smooth_map(self, x, k):
#         """
#         Smooth `x` using `k` nearest neighboring regions.
#
#         Parameters
#         ----------
#         x : 1D ndarray
#             Brain map
#         k : float
#             Number of nearest neighbors to include for smoothing
#
#         Returns
#         -------
#         x_smooth : 1D ndarray
#             Smoothed brain map
#
#         Notes
#         -----
#         Assumes `D` provided at runtime has been sorted.
#
#         """
#         jkn = self._index[:, :k]  # indices of k nearest neighbors
#         xkn = x[jkn]  # values of k nearest neighbors
#         dkn = self._D[:, :k]  # distances to k nearest neighbors
#         weights = self._kernel(dkn)  # distance-weighted kernel
#         # Kernel-weighted sum
#         return (weights * xkn).sum(axis=1) / weights.sum(axis=1)
#
#     def smooth_variogram(self, u, v, return_h=False):
#         """
#         Smooth a variogram.
#
#         Parameters
#         ----------
#         u : 1D ndarray
#             Pairwise distances, ie variogram x-coordinates
#         v : 1D ndarray
#             Squared differences, ie variogram y-coordinates
#         return_h : bool, default False
#             Return distances at which smoothed variogram is computed
#
#         Returns
#         -------
#         ndarray, shape (nh,)
#             Smoothed variogram samples
#         ndarray, shape (nh,)
#             Distances at which smoothed variogram was computed (returned if
#             `return_h` is True)
#
#         Raises
#         ------
#         ValueError : `u` and `v` are not identically sized
#
#         """
#         if len(u) != len(v):
#             raise ValueError("u and v must have same number of elements")
#
#         # Subtract each element of h from each pairwise distance `u`.
#         # Each row corresponds to a unique h.
#         du = np.abs(u - self._h[:, None])
#         w = np.exp(-np.square(2.68 * du / self._b) / 2)
#         denom = w.sum(axis=1)
#         wv = w * v[None, :]
#         num = wv.sum(axis=1)
#         output = num / denom
#         if not return_h:
#             return output
#         return output, self._h
#
#     def regress(self, x, y):
#         """
#         Linearly regress `x` onto `y`.
#
#         Parameters
#         ----------
#         x : 1D ndarray
#             Independent variable
#         y : 1D ndarray
#             Dependent variable
#
#         Returns
#         -------
#         alpha : float
#             Intercept term (offset parameter)
#         beta : float
#             Regression coefficient (scale parameter)
#         res : float
#             Sum of squared residuals
#
#         """
#         self._lm.fit(X=np.expand_dims(x, -1), y=y)
#         beta = self._lm.coef_.item()
#         alpha = self._lm.intercept_
#         ypred = self._lm.predict(np.expand_dims(x, -1))
#         res = np.sum(np.square(y-ypred))
#         return alpha, beta, res
#
#     def sample(self):
#         """
#         Randomly sample (without replacement) brain areas for variogram
#         computation.
#
#         Returns
#         -------
#         ndarray, shape (ns,)
#             Indices of randomly sampled areas
#
#         """
#         return np.random.choice(
#             a=self._nmap, size=self._ns, replace=False).astype(np.int32)
#
#     @property
#     def x(self):
#         """1D ndarray : brain map scalars """
#         if self._ismasked:
#             return np.ma.copy(self._x)
#         return np.copy(self._x)
#
#     @x.setter
#     def x(self, x):
#         self._ismasked = False
#         x_ = dataio(x)
#         check_map(x=x_)
#         mask = np.isnan(x_)
#         if mask.any():
#             self._ismasked = True
#             brain_map = np.ma.masked_array(data=x_, mask=mask)
#         else:
#             brain_map = x_
#         self._x = brain_map
#
#     @property
#     def D(self):
#         """ndarray or memmap, shape (N,N) : Pairwise distance matrix """
#         return np.copy(self._D)
#
#     @D.setter
#     def D(self, x):
#         x_ = dataio(x)
#         n = self._x.size
#         if x_.shape[0] != n:
#             raise ValueError(
#                 "D size along axis=0 must equal brain map size")
#         self._D = x_[:, 1:self._knn + 1]  # prevent self-coupling
#
#     @property
#     def index(self):
#         """ndarray or memmap : indexes used to sort each row of dist. matrix """
#         return np.copy(self._index)
#
#     @index.setter
#     def index(self, x):
#         x_ = dataio(x)
#         n = self._x.size
#         if x_.shape[0] != n:
#             raise ValueError(
#                 "index size along axis=0 must equal brain map size")
#         self._index = x_[:, 1:self._knn+1].astype(np.int32)
#
#     @property
#     def nmap(self):
#         """ int : length of brain map """
#         return self._nmap
#
#     @nmap.setter
#     def nmap(self, x):
#         self._nmap = int(x)
#
#     @property
#     def pv(self):
#         """ int : percentile of pairwise distances at which to truncate """
#         return self._pv
#
#     @pv.setter
#     def pv(self, x):
#         pv = check_pv(x)
#         self._pv = pv
#
#     @property
#     def deltas(self):
#         """ 1D ndarray or List[float] : proportions of nearest neighbors """
#         return self._deltas
#
#     @deltas.setter
#     def deltas(self, x):
#         check_deltas(deltas=x)
#         self._deltas = x
#
#     @property
#     def nh(self):
#         """ int : number of variogram distance intervals """
#         return self._nh
#
#     @nh.setter
#     def nh(self, x):
#         self._nh = x
#
#     @property
#     def kernel(self):
#         """ Callable : smoothing kernel function
#
#         Notes
#         -----
#         When setting kernel, use name of kernel as defined in ``config.py``.
#
#         """
#         return self._kernel
#
#     @kernel.setter
#     def kernel(self, x):
#         kernel_callable = check_kernel(x)
#         self._kernel = kernel_callable
#
#     @property
#     def resample(self):
#         """ bool : whether to resample surrogate map values from target maps """
#         return self._resample
#
#     @resample.setter
#     def resample(self, x):
#         if not isinstance(x, bool):
#             raise TypeError("expected bool, got {}".format(type(x)))
#         self._resample = x
#
#     @property
#     def knn(self):
#         """ int : number of nearest neighbors included in distance matrix """
#         return self._knn
#
#     @knn.setter
#     def knn(self, x):
#         if x > self._nmap:
#             raise ValueError('knn must be less than len(X)')
#         self._knn = int(x)
#
#     @property
#     def ns(self):
#         """ int : number of randomly sampled regions used to construct map """
#         return self._ns
#
#     @ns.setter
#     def ns(self, x):
#         self._ns = int(x)
#
#     @property
#     def b(self):
#         """ numeric : Gaussian kernel bandwidth """
#         return self._b
#
#     @b.setter
#     def b(self, x):
#         self._b = x
#
#     @property
#     def h(self):
#         """ 1D ndarray : distances at which variogram is evaluated """
#         return self._h
#
#     @h.setter
#     def h(self, x):
#         self._h = x


class SurrogateMaps(BaseEstimator):
    """ Spatial autocorrelation-preserving surrogate brain maps.

    Parameters
    ----------
    deltas : 1D ndarray or List[float], optional
        Proportion of neighbors to include for smoothing, in (0, 1]
        Default is [0.1,0.2,...,0.9].
    kernel : str, optional
        Kernel with which to smooth permuted maps:
          'gaussian' : Gaussian function.
          'exp' : Exponential decay function.
          'invdist' : Inverse distance.
          'uniform' : Uniform weights (distance independent).
        Default is 'exp'.
    pv : int, optional
        Percentile of the pairwise distance distribution at which to
        truncate during variogram fitting. Default is 25.
    nh : int, optional
        Number of uniformly spaced distances at which to compute variogram.
        Default is 25.
    resample : bool, optional
        Resample surrogate maps' values from target brain map.
        Default is False.
    b : float or None, optional
        Gaussian kernel bandwidth for variogram smoothing. If None, set to
        three times the spacing between variogram x-coordinates.
        Default is None.
    n_rep : int, optional
        Number of randomizations (i.e., surrogate maps). Default is 100.
    random_state : int or None, optional
        Random state. Default is None.

    See Also
    --------
    :class:`.SampledSurrogateMaps`
    :class:`.SpinPermutations`
    :class:`.MoranRandomization`

    Notes
    -----
    Passing resample=True preserves the distribution of values in the target
    map, with the possibility of worsening the simulated surrogate maps'
    variograms fits.

    """

    def __init__(self, deltas=None, kernel='exp', pv=25, nh=25, resample=False,
                 b=None, n_rep=100, random_state=None):

        self.deltas = np.linspace(0.1, 0.9, 9) if deltas is None else deltas
        check_deltas(deltas=self.deltas)

        self.resample = resample
        self.nh = nh
        self.pv = check_pv(pv)
        self.kernel = check_kernel(kernel)  # Smoothing kernel selection
        self.b = b
        self.n_rep = n_rep
        self.random_state = random_state

        self._rs = np.random.RandomState(self.random_state)

        # Linear regression model
        self._lm = LinearRegression(fit_intercept=True)

    def _check_distance(self, dist):
        d = dataio(dist)
        if not np.allclose(d, d.T):
            raise ValueError("Distance matrix must be symmetric")
        return d

    def _check_map(self, x):
        x_ = dataio(x)
        check_map(x=x_)
        brain_map = np.ma.masked_array(data=x_, mask=np.isnan(x_))
        if self._dist.shape[0] != brain_map.size:
            e = "Brain map must have dimensions consistent with distance matrix"
            e += "\nDistance matrix shape: {}".format(self._dist.shape)
            e += "\nBrain map size: {}".format(brain_map.size)
            raise ValueError(e)

        return brain_map

    def fit(self, dist):
        """ Prepare data for sorrugate map generation..

        Parameters
        ----------
        dist : filename or ndarray, shape (N,N)
            Pairwise (geodesic) distance matrix.

        Returns
        -------
        self : object
            Returns self.

        """

        self._dist = self._check_distance(dist)
        self.nmap = n = dist.shape[0]

        self._ikn = np.arange(n)[:, None]
        self._triu = np.triu_indices(n, k=1)  # upper triangular inds
        self._u = self._dist[self._triu]  # variogram X-coordinate
        # self._v = self.compute_variogram(self._x)  # variogram Y-coord

        # Get indices of pairs with u < pv'th percentile
        self._uidx = np.where(self._u < np.percentile(self._u, self.pv))[0]
        self._uisort = np.argsort(self._u[self._uidx])

        # Find sorted indices of first `kmax` elements of each row of dist.
        self._disort = np.argsort(self._dist, axis=-1)
        self._jkn = dict.fromkeys(self.deltas)
        self._dkn = dict.fromkeys(self.deltas)
        for delta in self.deltas:
            k = int(delta*n)
            # find index of k nearest neighbors for each area
            self._jkn[delta] = self._disort[:, 1:k+1]  # prevent self-coupling
            # find distance to k nearest neighbors for each area
            self._dkn[delta] = self._dist[(self._ikn, self._jkn[delta])]

        # Smoothed variogram and variogram _b
        utrunc = self._u[self._uidx]
        self._h = np.linspace(utrunc.min(), utrunc.max(), self.nh)
        if self.b is None:
            self.b = 3.*np.mean(self._h[1:] - self._h[:-1])

        return self

    def randomize(self, x, n_rep=None):
        """ Generate surrogate maps from `x`.

        Parameters
        ----------
        x : filename or 1D ndarray
            Target brain map
        n_rep : int or None, optional
            Number of surrogates maps to randomly generate. If None, use
            the default `n_rep`.

        Returns
        -------
        output : ndarray, shape = (n_rep, n_verts)
            Randomly generated map(s) with matched spatial autocorrelation.

        """

        x = self._check_map(x)
        nmap = x.size

        n_rep = self.n_rep if n_rep is None else n_rep

        v = self.compute_variogram(x)  # variogram Y-coord
        smvar = self.smooth_variogram(v)

        surrs = np.empty((n_rep, nmap))
        for i in range(n_rep):  # generate random maps

            xperm = self.permute_map(x)  # Randomly permute values
            res = dict.fromkeys(self.deltas)

            for delta in self.deltas:  # foreach neighborhood size
                # Smooth the permuted map using delta proportion of
                # neighbors to reintroduce spatial autocorrelation
                sm_xperm = self.smooth_map(x=xperm, delta=delta)

                # Calculate empirical variogram of the smoothed permuted map
                vperm = self.compute_variogram(sm_xperm)

                # Calculate smoothed variogram of the smoothed permuted map
                smvar_perm = self.smooth_variogram(vperm)

                # Fit linear regression btwn smoothed variograms
                res[delta] = self.regress(smvar_perm, smvar)

            alphas, betas, residuals = np.array(
                [res[d] for d in self.deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self.deltas[iopt]
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=xperm, delta=dopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * self._rs.randn(nmap))
            surrs[i] = surr

        if self.resample:  # resample values from empirical map
            sorted_map = np.sort(x)
            for i, surr in enumerate(surrs):
                ii = np.argsort(surr)
                np.put(surr, ii, sorted_map)

        return surrs.squeeze()

    def compute_variogram(self, x):
        """
        Compute variogram values (i.e., one-half squared pairwise differences).

        Parameters
        ----------
        x : 1D ndarray
            Brain map scalar array

        Returns
        -------
        v : ndarray, shape (N(N-1)/2,)
           Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2

        """
        diff_ij = np.subtract.outer(x, x)
        v = 0.5 * np.square(diff_ij)[self._triu]
        return v

    def permute_map(self, x):
        """
        Return randomly permuted brain map.

        Parameters
        ----------
        x : 1D masked ndarray
            Brain map scalars

        Returns
        -------
        1D ndarray
            Random permutation of target brain map

        """

        pidx = self._rs.permutation(x.size)
        return np.ma.masked_array(data=x.data[pidx], mask=x.mask[pidx])

    def smooth_map(self, x, delta):
        """
        Smooth `x` using `delta` proportion of nearest neighbors.

        Parameters
        ----------
        x : 1D ndarray
            Brain map scalars
        delta : float
            Proportion of neighbors to include for smoothing, in (0, 1)

        Returns
        -------
        1D ndarray
            Smoothed brain map

        """
        # Values of k nearest neighbors for each brain area
        xkn = x[self._jkn[delta]]
        weights = self.kernel(self._dkn[delta])  # Distance-weight kernel
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, v, return_h=False):
        """
        Smooth a variogram.

        Parameters
        ----------
        v : 1D ndarray
            Variogram values, i.e. 0.5 * (x_i - x_j) ^ 2
        return_h : bool, default False
            Return distances at which the smoothed variogram was computed

        Returns
        -------
        1D ndarray, shape (nh,)
            Smoothed variogram values
        1D ndarray, shape (nh,)
            Distances at which smoothed variogram was computed (returned only
            if `return_h` is True)

        Raises
        ------
        ValueError : `v` has unexpected size.

        """
        u = self._u[self._uidx]
        v = v[self._uidx]
        if len(u) != len(v):
            raise ValueError(
                "argument v: expected size {}, got {}".format(len(u), len(v)))
        # Subtract each h from each pairwise distance u
        # Each row corresponds to a unique h
        du = np.abs(u - self._h[:, None])
        w = np.exp(-np.square(2.68 * du / self.b) / 2)
        denom = w.sum(axis=1)
        wv = w * v[None, :]
        num = wv.sum(axis=1)
        output = num / denom
        if not return_h:
            return output
        return output, self._h

    def regress(self, x, y):
        """
        Linearly regress `x` onto `y`.

        Parameters
        ----------
        x : 1D ndarray
            Independent variable
        y : 1D ndarray
            Dependent variable

        Returns
        -------
        alpha : float
            Intercept term (offset parameter)
        beta : float
            Regression coefficient (scale parameter)
        res : float
            Sum of squared residuals

        """
        self._lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self._lm.coef_.item()
        alpha = self._lm.intercept_
        y_pred = self._lm.predict(X=np.expand_dims(x, -1))
        res = np.sum(np.square(y-y_pred))
        return alpha, beta, res

    @property
    def h(self):
        """ 1D ndarray : distances at which smoothed variogram is computed """
        return self._h


class SampledSurrogateMaps(BaseEstimator):
    """
    Spatial autocorrelation-preserving surrogate brain maps wih sampling.

    Parameters
    ----------
    ns : int, optional
        Take a subsample of `ns` rows from `D` when fitting variograms.
        Default is 500.
    deltas : 1D ndarray or List[float], optional
        Proportion of neighbors to include for smoothing, in (0, 1]
        Default is [0.1,0.2,...,0.9].
    kernel : str, optional
        Kernel with which to smooth permuted maps:
          'gaussian' : Gaussian function.
          'exp' : Exponential decay function.
          'invdist' : Inverse distance.
          'uniform' : Uniform weights (distance independent).
        Default is 'exp'.
    pv : int, optional
        Percentile of the pairwise distance distribution at which to
        truncate during variogram fitting. Default is 25.
    nh : int, optional
        Number of uniformly spaced distances at which to compute variogram.
        Default is 25.
    knn : int, optional
        Number of nearest regions to keep in the neighborhood of each region.
        Default is 1000.
    b : float or None, default None
        Gaussian kernel bandwidth for variogram smoothing. if None,
        three times the distance interval spacing is used.
    resample : bool, optional
        Resample surrogate maps' values from target brain map.
        Default is False.
    n_rep : int, optional
        Number of randomizations (i.e., surrogate maps). Default is 100.
    random_state : int or None, optional
        Random state. Default is None.
    verbose : bool, default False
        Print surrogate count each time new surrogate map created

    See Also
    --------
    :class:`.SurrogateMaps`
    :class:`.SpinPermutations`
    :class:`.MoranRandomization`

    Notes
    -----
    Passing resample=True will preserve the distribution of values in the
    target map, at the expense of worsening simulated surrogate maps'
    variograms fits. This worsening will increase as the empirical map
    more strongly deviates from normality.

    """

    def __init__(self, ns=500, pv=70, nh=25, knn=1000, b=None, deltas=None,
                 kernel='exp', resample=False, n_rep=100, random_state=None,
                 verbose=False):

        self.deltas = np.arange(0.3, 1., 0.2) if deltas is None else deltas
        check_deltas(deltas=self.deltas)

        self.ns = int(ns)
        self.resample = resample
        self.nh = nh
        self.pv = check_pv(pv)
        self.kernel = check_kernel(kernel)  # Smoothing kernel selection
        self.knn = knn
        self.b = b
        self.verbose = verbose
        self.n_rep = n_rep
        self.random_state = random_state

        self._rs = np.random.RandomState(self.random_state)

        # Linear regression model
        self._lm = LinearRegression(fit_intercept=True)

    def _check_distance(self, dist):
        dist = dataio(dist)
        return dist[:, 1:self.knn + 1]  # prevent self-coupling

    def _check_index(self, index):
        index = dataio(index)
        return index[:, 1:self.knn + 1].astype(np.int32)

    def _check_map(self, x):
        self._ismasked = False
        x = dataio(x)
        check_map(x=x)
        mask = np.isnan(x)
        if mask.any():
            self._ismasked = True
            return np.ma.masked_array(data=x, mask=mask)
        return x

    def fit(self, dist, index):
        """ Prepare data for surrogate map generation.

        Parameters
        ----------
        dist : ndarray or memmap, shape (N,N)
            Pairwise distance matrix. Each row of `dist` should be sorted.
            Indices used to sort each row are passed to through the `index`
            argument. See :func:`brainspace.variogram.txt2memmap`.
        index : filename or ndarray or memmap, shape(N,N)
            Indices used to sort each row of `dist`.

        Returns
        -------
        self : object
            Returns self.

        """

        self._dist = self._check_distance(dist)
        self._index = self._check_index(index)
        if self._dist.shape != self._index.shape:
            raise ValueError('Dimensions of distanace and index matrices are '
                             'inconsistent: distance shape = {}, index '
                             'shape = {}.'.format(self._dist.shape,
                                                  self._index.shape))

        n = self._dist.shape[0]
        self._ikn = np.arange(n)[:, None]
        self._dmax = np.percentile(self._dist, self.pv)
        self._h = np.linspace(self._dist.min(), self._dmax, self.nh)
        if not self.b:
            self.b = 3 * (self._h[1] - self._h[0])

        return self

    def randomize(self, x, n_rep=None):
        """ Generate surrogate maps from `x`.

        Parameters
        ----------
        x : filename or 1D ndarray
            Target brain map
        n_rep : int or None, optional
            Number of surrogates maps to randomly generate. If None, use
            the default `n_rep`.

        Returns
        -------
        output : ndarray, shape = (n_rep, n_verts)
            Randomly generated map(s) with matched spatial autocorrelation.

        """

        x = self._check_map(x)
        n = x.size

        if n != self._dist.shape[0]:
            raise ValueError("Size of distance matrix along axis=0 must equal "
                             "brain map size.")

        n_rep = self.n_rep if n_rep is None else n_rep

        if self.verbose:
            print("Generating {} maps...".format(n_rep))

        surrs = np.empty((n_rep, n))
        for i in range(n_rep):  # generate random maps
            if self.verbose:
                print(i+1)

            # Randomly permute map
            x_perm = self.permute_map(x)

            # Randomly select subset of regions to use for variograms
            idx = self.sample(n)

            # Compute empirical variogram
            v = self.compute_variogram(x, idx)

            # Variogram ordinates; use nearest neighbors because local effect
            u = self._dist[idx, :]
            uidx = np.where(u < self._dmax)

            # Smooth empirical variogram
            smvar, u0 = self.smooth_variogram(u[uidx], v[uidx], return_h=True)

            res = dict.fromkeys(self.deltas)
            for d in self.deltas:  # foreach neighborhood size

                k = int(d * self.knn)

                # Smooth the permuted map using k nearest neighbors to
                # reintroduce spatial autocorrelation
                sm_xperm = self.smooth_map(x=x_perm, k=k)

                # Calculate variogram values for the smoothed permuted map
                vperm = self.compute_variogram(sm_xperm, idx)

                # Calculate smoothed variogram of the smoothed permuted map
                smvar_perm = self.smooth_variogram(u[uidx], vperm[uidx])

                # Fit linear regression btwn smoothed variograms
                res[d] = self.regress(smvar_perm, smvar)

            alphas, betas, residuals = np.array(
                [res[d] for d in self.deltas]).T

            # Select best-fit model and regression parameters
            iopt = np.argmin(residuals)
            dopt = self.deltas[iopt]
            kopt = int(dopt * self.knn)
            aopt = alphas[iopt]
            bopt = betas[iopt]

            # Transform and smooth permuted map using best-fit parameters
            sm_xperm_best = self.smooth_map(x=x_perm, k=kopt)
            surr = (np.sqrt(np.abs(bopt)) * sm_xperm_best +
                    np.sqrt(np.abs(aopt)) * self._rs.randn(n))
            surrs[i] = surr

        if self.resample:  # resample values from empirical map
            sorted_map = np.sort(x)
            for i, surr in enumerate(surrs):
                ii = np.argsort(surr)
                np.put(surr, ii, sorted_map)

        if self._ismasked:
            return np.ma.masked_array(
                data=surrs, mask=np.isnan(surrs)).squeeze()
        return surrs.squeeze()

    def compute_variogram(self, x, idx):
        """
        Compute variogram of `x` using pairs of regions indexed by `idx`.

        Parameters
        ----------
        x : 1D ndarray
            Brain map
        idx : ndarray[int], shape (ns,)
            Indices of randomly sampled brain regions

        Returns
        -------
        v : ndarray, shape (ns,ns)
            Variogram y-coordinates, i.e. 0.5 * (x_i - x_j) ^ 2,
            for i,j in idx

        """

        diff_ij = x[idx][:, None] - x[self._index[idx, :]]
        return 0.5 * np.square(diff_ij)

    def permute_map(self, x):
        """
        Return a random permutation of `x`.

        Parameters
        ----------
        x : 1D ndarray
            Brain map

        Returns
        -------
        1D ndarray
            Random permutation of target brain map

        """

        pidx = self._rs.permutation(x.size)
        if self._ismasked:
            return np.ma.masked_array(data=x.data[pidx], mask=x.mask[pidx])
        return x[pidx]

    def smooth_map(self, x, k):
        """
        Smooth `x` using `k` nearest neighboring regions.

        Parameters
        ----------
        x : 1D ndarray
            Brain map
        k : float
            Number of nearest neighbors to include for smoothing

        Returns
        -------
        x_smooth : 1D ndarray
            Smoothed brain map

        Notes
        -----
        Assumes `dist` provided at runtime has been sorted.

        """
        jkn = self._index[:, :k]  # indices of k nearest neighbors
        xkn = x[jkn]  # values of k nearest neighbors
        dkn = self._dist[:, :k]  # distances to k nearest neighbors
        weights = self.kernel(dkn)  # distance-weighted kernel
        return (weights * xkn).sum(axis=1) / weights.sum(axis=1)

    def smooth_variogram(self, u, v, return_h=False):
        """
        Smooth a variogram.

        Parameters
        ----------
        u : 1D ndarray
            Pairwise distances, ie variogram x-coordinates
        v : 1D ndarray
            Squared differences, ie variogram y-coordinates
        return_h : bool, default False
            Return distances at which smoothed variogram is computed

        Returns
        -------
        ndarray, shape (nh,)
            Smoothed variogram samples
        ndarray, shape (nh,)
            Distances at which smoothed variogram was computed (returned if
            `return_h` is True)

        Raises
        ------
        ValueError : `u` and `v` are not identically sized

        """
        if len(u) != len(v):
            raise ValueError("u and v must have same number of elements")

        # Subtract each element of h from each pairwise distance `u`.
        # Each row corresponds to a unique h.
        du = np.abs(u - self._h[:, None])
        w = np.exp(-np.square(2.68 * du / self.b) / 2)
        denom = w.sum(axis=1)
        wv = w * v[None, :]
        num = wv.sum(axis=1)
        output = num / denom
        if not return_h:
            return output
        return output, self._h

    def regress(self, x, y):
        """
        Linearly regress `x` onto `y`.

        Parameters
        ----------
        x : 1D ndarray
            Independent variable
        y : 1D ndarray
            Dependent variable

        Returns
        -------
        alpha : float
            Intercept term (offset parameter)
        beta : float
            Regression coefficient (scale parameter)
        res : float
            Sum of squared residuals

        """
        self._lm.fit(X=np.expand_dims(x, -1), y=y)
        beta = self._lm.coef_.item()
        alpha = self._lm.intercept_
        ypred = self._lm.predict(np.expand_dims(x, -1))
        res = np.sum(np.square(y-ypred))
        return alpha, beta, res

    def sample(self, n):
        """
        Randomly sample (without replacement) brain areas for variogram
        computation.

        Returns
        -------
        ndarray, shape (ns,)
            Indices of randomly sampled areas

        """
        return self._rs.choice(a=n, size=self.ns,
                               replace=False).astype(np.int32)

    @property
    def h(self):
        """ 1D ndarray : distances at which variogram is evaluated """
        return self._h
