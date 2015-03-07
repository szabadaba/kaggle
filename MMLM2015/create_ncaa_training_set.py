__author__ = 'Steven Szabados'

import pandas as ps
import numpy as np
import sys


season_detail_filename = 'data/regular_season_detailed_results.csv'

class CreateNCAATrainingSet:
    def __init__(self):
        # load seson data
        data = ps.io.parsers.read_csv(season_detail_filename)

        # break up data into seasons
        s_data = []
        for season in data.season.unique():
            s_data.append(data[data.season == season])


class CreatePreGameStats:
    def __init__(self, game, season):
        MIN_DAYS = 30

        # create first game day
        daynum1 = min(season.daynum)

        # do we have enough days?
        if (game.daynum - daynum1) > MIN_DAYS:
            eval_games = season[season.daynum < game.daynum]

        # self.team = team
        # self.pts_rank = pts_rank
        # self.avg_pts_for = avg_pts_for
        # self.avg_point_against = avg_point_against
        # self.ft_perc = ft_perc
        # self.fg_perc = fg_perc
        # self.tp_perc = tp_perc

class CreateWeightedPtsRanking:
    def __init__(self, games):
        # get all the teams
        teams = []
        for row in games.iterrows():
            game = row[1]
            if game.wteam not in teams:
                teams.append(game.team)
            if game.lteam not in teams:
                teams.append(game.team)

        # create pts matrix
        point_mtx = np.zeros((len(teams), len(teams)))
        numb_games = np.zeros(len(teams))

        # set max daynum
        max_daynum = max(games.daynum)

        # create points mtx
        for row in games.iterrows():
            game = row[1]
            # get idx's
            w_idx = teams.index(game.wteam)
            l_idx = teams.index(game.lteam)

            # create dampen factor
            dampen_factor = (1.0/(max_daynum - game.daynum + 1))

            # add game to count
            numb_games[w_idx] += 1*dampen_factor
            numb_games[l_idx] += 1*dampen_factor

            # add points to mtx
            point_mtx[w_idx, l_idx] += dampen_factor*float(row[1].wscore)/(float(row[1].wscore + row[1].lscore))
            point_mtx[l_idx, w_idx] += dampen_factor*float(row[1].lscore)/float((row[1].wscore + row[1].lscore))

        # scale points matrix
        d = np.diag(1.0/numb_games)
        point_mtx_scale = np.dot(d, self.point_mtx)

        # make sure the matrix is not singular
        if np.linalg.cond(point_mtx_scale) < 1/sys.float_info.epsilon:
            v = np.asarray(np.linalg.eig(point_mtx_scale)[1].real).tolist
            w_team_rank = v(teams.index(game.wteam))






class NCAAGameTrainingInstance:
    def __init__(self, team1_pgs, team2_pgs, outcome):
        self.team1_pgs = team1_pgs
        self.team2_pgs = team2_pgs
        self.outcome = outcome