__author__ = 'Steve'
import pickle

import load_mmlm_data as ld
import load_season_data as ls
import aggregate_ncaa_season_data as ag
import create_ncaa_training_set as nc

import analyze_cbb_season as an

t = nc.CreateNCAATrainingSet()

# save data
pickle.dump(t.formatted_data, open("training_set_X.p", "wb"))
pickle.dump(t.truth_label, open("training_set_y.p", "wb"))


# t = ld.load_data()
# t = ls.LoadSeasonData()

# r = an.point_matrix(t.seasons[-2], t.teams)
# r = ag.RankByPoints(t.seasons[-1], t.teams)
x = 1

