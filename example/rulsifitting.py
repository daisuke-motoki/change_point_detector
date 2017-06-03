import cPickle as pickle
import peakutils
import numpy as np

from change_point_detector.density_ratio_estimator import DRChangeRateEstimator

with open("../test/data/test.pkl") as file_:
    data = pickle.load(file_)

detector = DRChangeRateEstimator(sliding_window=3,
                                 pside_len=50,
                                 cside_len=50,
                                 mergin=-1,
                                 trow_offset=0,
                                 tcol_offset=0)
detector.build(estimation_method="RuLSIFitting",
               options=detector.RuLSIF_OPTION)
change_rates = detector.transform(data["y"])
change_rates = np.nan_to_num(change_rates)
