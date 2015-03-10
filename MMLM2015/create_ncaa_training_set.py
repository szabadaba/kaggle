__author__ = 'Steven Szabados'

from joblib import Parallel, delayed
import multiprocessing
import pandas as ps
import numpy as np
from math import sqrt
import sys


season_detail_filename = 'data/regular_season_detailed_results.csv'

class CreateNCAATrainingSet:
    def __init__(self):
        # load seson data
        data = ps.io.parsers.read_csv(season_detail_filename)
        self.game_stats = []

        # break up data into seasons
        s_data = []
        for season in data.season.unique():
            s_data.append(data[data.season == season])

        for season in s_data:
            self.game_stats.append(CreatePreGameStats())
            print season.irow(0).season
            for game in season.iterrows():
                self.game_stats[-1].add_game(game[1], season)

            s_win_prob = np.asarray(self.game_stats[-1].w_team_prob)
            # s_win_prob2 = np.asarray(self.game_stats[-1].w_team_prob2)
            print float(sum(s_win_prob > 1.0))/len(s_win_prob)
            # print float(sum(s_win_prob2 > 1.0))/len(s_win_prob2)

        # for game in s_data[0].iterrows():
        #     self.game_stats.add_game(game[1], s_data[0])


class CreatePreGameStats:
    def __init__(self):
        self.MIN_DAYS = 30
        self.pts_rank_results = []
        self.ptr_rank_daynum = []
        self.w_team_prob = []
        # self.w_team_prob2 = []

    def add_game(self, game, season):

        # create first game day
        daynum1 = min(season.daynum)

        # do we have enough days?
        if (game.daynum - daynum1) > self.MIN_DAYS:
            # create games
            eval_games = season[season.daynum < game.daynum]

            # do we not have this ranking?
            if game.daynum not in self.ptr_rank_daynum:
                pts_rank = CreateWeightedPtsRanking(eval_games)
                if pts_rank.success:
                    self.pts_rank_results.append(pts_rank)
                    self.ptr_rank_daynum.append(game.daynum)

            # can we create the points ranking?
            if game.daynum in self.ptr_rank_daynum:
                pts_rank = self.pts_rank_results[self.ptr_rank_daynum.index(game.daynum)]
                # did we get a ranking for both teams:
                if game.wteam in pts_rank.teams and game.lteam in pts_rank.teams:
                    w_team_rank = pts_rank.team_rank[pts_rank.teams.index(game.wteam)]
                    l_team_rank = pts_rank.team_rank[pts_rank.teams.index(game.lteam)]
                    # print float(w_team_rank/l_team_rank)
                    self.w_team_prob.append(float(w_team_rank/l_team_rank))

                    # w_team_w_perc = float(sum(eval_games.wteam == game.wteam))/float((sum(eval_games.wteam == game.wteam) + sum(eval_games.lteam == game.wteam)))
                    # l_team_w_perc = float(sum(eval_games.wteam == game.lteam))/float((sum(eval_games.wteam == game.lteam) + sum(eval_games.lteam == game.lteam)))
                    # self.w_team_prob2.append(float((w_team_w_perc + 0.5)/(l_team_w_perc + 0.5)))




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
                teams.append(game.wteam)
            if game.lteam not in teams:
                teams.append(game.lteam)

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
            dampen_factor = sqrt((1.0/(max_daynum - game.daynum + 1)))
            # dampen_factor = 1

            # add game to count
            numb_games[w_idx] += 1*dampen_factor
            numb_games[l_idx] += 1*dampen_factor

            # add points to mtx
            point_mtx[w_idx, l_idx] += dampen_factor*float(row[1].wscore)/(float(row[1].wscore + row[1].lscore))
            point_mtx[l_idx, w_idx] += dampen_factor*float(row[1].lscore)/float((row[1].wscore + row[1].lscore))
            # point_mtx[w_idx, l_idx] += dampen_factor*float(row[1].wscore)
            # point_mtx[l_idx, w_idx] += dampen_factor*float(row[1].lscore)

        # scale points matrix
        d = np.diag(1.0/numb_games)
        point_mtx_scale = np.dot(d,point_mtx)

        # make sure the matrix is not singular
        if np.linalg.cond(point_mtx_scale) < 1/sys.float_info.epsilon:
            V = np.linalg.eig(point_mtx_scale)[1]
            self.team_rank = abs(np.asarray(V[:, 0].real).transpose()).tolist()
            self.teams = teams
            self.success = True
        else:
            self.success = False






class NCAAGameTrainingInstance:
    def __init__(self, team1_pgs, team2_pgs, outcome):
        self.team1_pgs = team1_pgs
        self.team2_pgs = team2_pgs
        self.outcome = outcome