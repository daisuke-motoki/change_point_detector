# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.utils.extmath import randomized_svd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances


class DRChangeRateEstimator(object):
    """ density ratio change rate estimator
    """
    MEAN_OPTION = {
        "feature_extraction": {
            "name": "mean",
            "param": {}
        }
    }
    SVD_OPTION = {
        "feature_extraction": {
            "name": "svd",
            "param": {
                "n_components": 2,
                "n_iter": 5
            }
        }
    }
    RuLSIF_OPTION = {
        "param": {
            "lambda": 0.1,
            "beta": 0.2,
            "gamma": 0.1,
            "validate": "none"
        }
    }

    def __init__(self,
                 sliding_window=3,
                 pside_len=4,
                 cside_len=4,
                 mergin=0,
                 trow_offset=0,
                 tcol_offset=0):
        """ init
        Args:
            sliding_window: int:
            pside_len: int(>0): past side length
            cside_len: int(>0): current side length
            mergin: int(<pside_len):
                mergin btween past side and current side
            offset: int(0=<t_offset<=cside_len): current time offset
        """
        self.sliding_window = sliding_window
        self.pside_len = pside_len
        self.cside_len = cside_len
        self.mergin = mergin
        self.trow_offset = trow_offset
        self.tcol_offset = tcol_offset
        self._estimation_method = None

    def build(self,
              estimation_method="von_mises_fisher",
              options=MEAN_OPTION):
        """ build
        Args:
            estimation_method: string:
                "von_mises_fisher":
                "RuLSIFitting":
        """
        if estimation_method == "von_mises_fisher":
            # set feature extractor
            extractor_options = options.get("feature_extraction", dict())
            name = extractor_options.get("name")
            params = extractor_options.get("param")
            if name == "mean":
                extractor = MeanVectorExtractor()
            elif name == "svd":
                extractor = SVDExtractor(
                    n_components=params["n_components"],
                    n_iter=params["n_iter"]
                )
            else:
                raise TypeError("unknown extractor {}".format(name))
            # set estimation method
            self._estimation_method = VonMisesFisherFitting(extractor)

        elif estimation_method == "RuLSIFitting":
            # set estimation method
            params = options.get("param")
            self._estimation_method = RuLSIFitting(
                params["gamma"], params["lambda"], params["beta"],
                params["validate"]
            )

        else:
            raise TypeError("unknown method name {}".format(estimation_method))

    def transform(self, X, destination="forward_backward"):
        """ transform
        Args:
            X: numpy array 1D: time series or other 1D features
               or
               numpy array 2D: time series subsequence or other 2D features
            destination: string:
                "forward":
                "backward":
                "forward_backward":
        Returns:
            change_rates: numpy array 1D:
        """
        nT = len(X)
        # check input data shape and convert them to 2D if they are 1D
        if len(X.shape) == 1:
            X = self.get_subsequence(X, self.sliding_window)
        elif len(X.shape) > 2:
            raise ValueError("input data shape should be 1 or 2d.")

        if X.shape[1] != self.sliding_window:
            raise ValueError("input shape does not match sliding window.")

        change_rates = np.zeros(nT)
        for t in xrange(nT):
            pX, cX = self._get_pcX(t, X)
            if pX is None or cX is None:
                change_rates[t] = None
            else:
                if destination == "forward":
                    change_rates[t] = self._estimation_method(pX, cX)
                elif destination == "backward":
                    if t+1 < nT:
                        change_rates[t+1] = self._estimation_method(cX, pX)
                else:
                    change_rates[t] += self._estimation_method(pX, cX)/2.0
                    if t+1 < nT:
                        change_rates[t+1] += self._estimation_method(cX, pX)/2.0

        return change_rates

    def _get_pcX(self, t, X):
        """ get past and current side X
        Args:
            t: int: time index of X
            X: numpy array 2D:
        Returns:
            pX: numpy array 2D:
            cX: numpy array 2D:
        """
        cindex = t - self.tcol_offset - self.trow_offset
        if cindex + self.cside_len > len(X) or cindex < 0:
            cX = None
        else:
            cX = X[cindex:cindex+self.cside_len]
        pindex = cindex - self.pside_len - self.mergin
        if pindex < 0:
            pX = None
        else:
            pX = X[pindex:pindex+self.pside_len]
        return pX, cX

    @staticmethod
    def get_subsequence(X, m):
        """ get subsequence
        Args:
            X: numpy array 1D:
            m: int: sliding_window
        Returns:
            subsequences: numpy array 2D:
        """
        n = len(X) - m + 1
        subsequences = np.zeros((n, m))
        for i in xrange(n):
            subsequences[i] = X[i:i+m]
        return subsequences


class VonMisesFisherFitting(object):
    """ Von Mises Fiser Method Fitting Class
    """
    def __init__(self, extractor):
        """ init
        """
        self._feature_extractor = extractor

    def __call__(self, pX, cX):
        """ call method
        Args:
            pX: numpy array 2D: past side subsequences
            cX: numpy array 2D: current side subsequences
        """
        return self.estimate(pX, cX)

    def estimate(self, pX, cX):
        """ change rate estimation method by using von mises fisher
            distribution
        Args:
            pX: numpy array 2D: past side subsequences
            cX: numpy array 2D: current side subsequences
        """
        # feature extraction from 2D matrix
        if self._feature_extractor is None:
            raise TypeError("Feature extractor is not assigned.")
        u = self._feature_extractor.extract(pX)
        q = self._feature_extractor.extract(cX)
        if len(u.shape) > 1:
            return (1. - np.linalg.norm((u.T).dot(q), 2))
        else:
            return (1. - (u.T).dot(q))


class RuLSIFitting(object):
    """ relative unconstrained least-squares importance fitting
        Reference
            * Anomaly Detection and Change Detection,
              T.Ide and M.Sugiyama, Kodansha Ltd.
            * Change-point detection in time-series data by relative
              density-ratio estimation, Song Liu, et.al.
              Neural Networks, volume 43, July 2013, Pages 72-83
    """
    def __init__(self, gamma, lam, beta, validate="none"):
        """ init
        Args:
            gamma: float: parameter of rbf kernel
            lam: float: parameter of regularization
            beta: float: parameter of relative pearson divergence (0~1)
            validate: string: times of validation
                "every": validate every change rate
                "once": validate only once data set
                "none": no validation (use initial parameters)
        """
        self.gamma = gamma
        self.lam = lam
        self.beta = beta
        self.validate = validate
        self.is_validated = False

        self.nfold = 5
        self.list_lambda = [
            0.0001, 0.001, 0.1, 1.0, 10.0
        ]
        self.list_sigma = [
            0.6, 0.8, 1.0, 1.2, 1.4
        ]

    def get_gammas(self, X, Y):
        """ get gammas
        Args:
            X: np array:
            Y: np array:
        Returns:
            gammas: np array:
                1/(2*sigma^2) where sigma is multipled by median distance
        """
        S = np.concatenate((X, Y))
        dists = euclidean_distances(S, S)
        med_dist = np.median(dists)
        sigmas = med_dist*np.array(self.list_sigma)
        gammas = np.array([1.0/(2.0*sigma*sigma) for sigma in sigmas])
        return gammas

    def __call__(self, pX, cX):
        """ call method
        Args:
            pX: numpy array 2D: past side subsequences
            cX: numpy array 2D: current side subsequences
        """
        return self.estimate(pX, cX)

    def estimate(self, pX, cX):
        """ change rate estimation method by using von mises fisher
            distribution
        Args:
            pX: numpy array 2D: past side subsequences
            cX: numpy array 2D: current side subsequences
        """
        # cross validation
        if(self.validate == "every" or
           (self.validate == "once" and self.is_validated is False)):
            list_gamma = self.get_gammas(pX, cX)
            self.cross_validation(
                pX, cX, self.nfold, self.list_lambda, list_gamma
            )

        # estimate change rate
        phi_x = rbf_kernel(pX, pX, self.gamma)
        phi_y = rbf_kernel(pX, cX, self.gamma)
        theta = self.calc_theta(phi_x, phi_y, self.lam, self.beta)
        if theta is None:
            return None
        rPE = self.calc_pe(theta, phi_x, phi_y)

        return rPE

    def calc_theta(self, phi_x, phi_y, lam, beta):
        """ calculate theta
        Args:
            phi_x:
            phi_y:
            lam:
            beta:
        Returns:
            theta:
        """
        nx = phi_x.shape[1]
        ny = phi_y.shape[1]
        h = phi_x.mean(axis=1)
        G_1 = beta/nx*self._product_sum(phi_x, phi_x)
        G_2 = (1.0 - beta)/ny*self._product_sum(phi_y, phi_y)
        G_beta = G_1 + G_2 + lam*np.eye(len(G_1))

        if np.linalg.matrix_rank(G_beta) != len(G_beta):
            return None
        theta = np.linalg.solve(G_beta, h)
        return theta

    def calc_pe(self, theta, phi_x, phi_y):
        """ calculate pearson divergence
        Args:
            theta:
            phi_x:
            phi_y:
        Returns:
            :
        """
        g_x = theta.dot(phi_x)
        g_y = theta.dot(phi_y)
        term1 = -1.0*self.beta/2.0*(g_x**2).mean()
        term2 = -1.0*(1.0-self.beta)/2.0*(g_y**2).mean()
        term3 = g_x.mean()

        return term1 + term2 + term3 - 1.0/2.0

    def cross_validation(self,
                         X,
                         Y,
                         nfold,
                         list_lambda,
                         list_gamma,
                         use_flat=False):
        """ cross validation
        Args:
            X:
            Y:
            nfold:
            list_lambda:
            list_gamma:
        """
        kf = KFold(n_splits=nfold)
        if use_flat:
            nX = X.flatten()
            nY = Y.flatten()
        else:
            nX = X
            nY = Y

        cv_score = np.zeros((len(list_gamma), len(list_lambda)))
        for ig, test_gamma in enumerate(list_gamma):
            phi_x = rbf_kernel(nX, nX, test_gamma)
            phi_y = rbf_kernel(nX, nY, test_gamma)
            cv_score_k = np.zeros((nfold, len(list_lambda)))
            for k, (ix, iy) in enumerate(zip(kf.split(phi_x.T),
                                             kf.split(phi_y.T))):
                train_ix, test_ix = ix
                train_iy, test_iy = iy
                for il, test_lambda in enumerate(list_lambda):
                    theta_k = self.calc_theta(phi_x[:, train_ix],
                                              phi_y[:, train_iy],
                                              test_lambda,
                                              self.beta)
                    J_k = self.calc_pe(
                        theta_k, phi_x[:, test_ix], phi_y[:, test_iy]
                    )
                    cv_score_k[k, il] = J_k
            cv_score[ig] = cv_score_k.mean(axis=0)
        answer_gi = cv_score.argmin(axis=0)[0]
        answer_li = cv_score.argmin(axis=1)[0]
        self.gamma = list_gamma[answer_gi]
        self.lam = list_lambda[answer_li]
        self.is_validated = True

    @staticmethod
    def _product_sum(phi1, phi2):
        return phi1.dot(phi2.T)


class Extractor(object):
    """ Extractor Base
    """
    __metaclass__ = ABCMeta

    def __call__(self, X):
        """ call
        Args:
            X: numpy array 2D:
        """
        return self.extract(X)

    @abstractmethod
    def extract(self, X):
        """ extract
        Args:
            X: numpy array 2D:
        """
        pass


class MeanVectorExtractor(Extractor):
    """ Mean Vector Extractor Class
    """
    def extract(self, X):
        """ extract
        Args:
            X: numpy array 2D:
        """
        m = X.sum(axis=0)/len(X)
        return m/np.sqrt(m.dot(m))


class SVDExtractor(Extractor):
    """ Singlar Value Decomposition Extractor Class
    """
    def __init__(self, n_components=2, n_iter=5):
        """ init
        """
        self.n_components = n_components
        self.n_iter = n_iter

    def extract(self, X):
        """ extract
        Args:
            X: numpy array 2D:
        """
        U, Sigma, VT = randomized_svd(
            X, n_components=self.n_components, n_iter=self.n_iter
        )
        return U
