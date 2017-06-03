import numpy as np


class TestDataSet(object):
    """ Test Data Set Generator Class
    """
    def get_step_series(self, widths, means, variances):
        """ get step series
        Args:
            widths: list: width of steps
                ex) [10, 10, 10]
            means: list: mean of steps
                ex) [1, 2, 3]
            variances: list: gaussian variance of steps
                ex) [0.5, 0.4, 0.3]
        Returns:
            times: numpy array 1D:
        """
        times = list()
        for i in xrange(len(widths)):
            times += (
                np.random.normal(
                    means[i], variances[i], int(widths[i])
                ).tolist()
            )
        return np.array(times, dtype=float)

    def get_slope_series(self, widths, gradients, variances, intercept=0.):
        """ get slope series
        Args:
            widths: list: width of steps
            gradients: list: a slope of function
            variances: list: gaussian variance of steps
            intercept: float: start point of series
        Returns:
            times: numpy array 1D:
        """
        times = [intercept]
        for i in xrange(len(widths)):
            slope_func = np.array(
                [gradients[i]*(t+1)+times[-1] for t in xrange(widths[i])]
            )
            noises = np.random.normal(0., variances[i], int(widths[i]))
            times += (slope_func + noises).tolist()
        del times[0]
        return np.array(times, dtype=float)
