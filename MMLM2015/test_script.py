__author__ = 'Steve'

import load_mmlm_data as ld
import analyze_cbb_season as an

t = ld.load_data()

r = an.point_matrix(t.seasons[0])
x = 1