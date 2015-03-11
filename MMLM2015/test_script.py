__author__ = 'Steve'
import pickle

import load_mmlm_data as ld
import load_season_data as ls
import aggregate_ncaa_season_data as ag
import create_ncaa_training_set as nc
import numpy as np
from sklearn import svm
import matplotlib as mp

import analyze_cbb_season as an

if 1:
    t = nc.CreateNCAATrainingSet()

    # save data
    pickle.dump(t.formatted_data, open("training_set_X.p", "wb"))
    pickle.dump(t.truth_label, open("training_set_y.p", "wb"))
else:
    X = np.asarray(pickle.load(open("training_set_X.p", "rb")))
    y = np.asarray(pickle.load(open("training_set_y.p", "rb"))).astype(np.int)

# clf = svm.SVC()
# clf.fit(X, y)

# t = ld.load_data()
# t = ls.LoadSeasonData()

# r = an.point_matrix(t.seasons[-2], t.teams)
# r = ag.RankByPoints(t.seasons[-1], t.teams)
x = 1

