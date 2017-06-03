import cPickle as pickle
import numpy as np

from change_point_detector.density_ratio_estimator import DRChangeRateEstimator

with open("../test/data/test.pkl") as file_:
    data = pickle.load(file_)

detector = DRChangeRateEstimator(sliding_window=5,
                                 pside_len=5,
                                 cside_len=5,
                                 mergin=-2,
                                 trow_offset=3,
                                 tcol_offset=3)
detector.build(estimation_method="von_mises_fisher",
               options=detector.SVD_OPTION)
change_rates = detector.transform(data["y"])
