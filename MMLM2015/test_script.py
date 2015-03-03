__author__ = 'Steve'

import load_mmlm_data as ld
import load_season_data as ls
import aggregate_ncaa_season_data as ag

import analyze_cbb_season as an

# t = ld.load_data()
t = ls.LoadSeasonData()

# r = an.point_matrix(t.seasons[-2], t.teams)
r = ag.RankByPoints(t.seasons[-1], t.teams)
x = 1