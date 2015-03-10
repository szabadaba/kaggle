__author__ = 'Steven Szabados'

# from joblib import Parallel, delayed
# import multiprocessing
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
        self.training_set = []
        self.formatted_data = []
        self.truth_label = []

        # break up data into seasons
        s_data = []
        # for season in [2014]:
        for season in data.season.unique():
            s_data.append(data[data.season == season])

        for season in s_data:
            self.game_stats.append(CreatePreGameStats())
            print season.irow(0).season
            for game in season.iterrows():
                self.game_stats[-1].add_game(game[1], season, self.training_set)

        self.aggregate_training_set()

            # s_win_prob = np.asarray(self.game_stats[-1].w_team_prob)
            # s_win_prob2 = np.asarray(self.game_stats[-1].w_team_prob2)
            # print float(sum(s_win_prob > 1.0))/len(s_win_prob)
            # print float(sum(s_win_prob2 > 1.0))/len(s_win_prob2)

        # for game in s_data[0].iterrows():
        #     self.game_stats.add_game(game[1], s_data[0])

    def aggregate_training_set(self):

        for te in self.training_set:
            self.formatted_data.append([te.w_team_pts_rank/te.l_team_pts_rank,
                                        te.w_fgp/te.l_fgp,
                                        te.w_ftp/te.l_ftp,
                                        te.w_orpg/te.l_orpg,
                                        te.w_drpg/te.l_drpg])
            self.truth_label.append(1)
            self.formatted_data.append([te.l_team_pts_rank/te.w_team_pts_rank,
                                        te.l_fgp/te.w_fgp,
                                        te.l_ftp/te.w_ftp,
                                        te.l_orpg/te.w_orpg,
                                        te.l_drpg/te.w_drpg])
            self.truth_label.append(0)



class CreatePreGameStats:
    def __init__(self):
        self.MIN_DAYS = 50
        self.pts_rank_results = []
        self.ptr_rank_daynum = []

    def add_game(self, game, season, training_set):

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
                    # set pts rank
                    w_team_rank = pts_rank.team_rank[pts_rank.teams.index(game.wteam)]
                    l_team_rank = pts_rank.team_rank[pts_rank.teams.index(game.lteam)]

                    # create training instance
                    training_set.append(NCAAGameTrainingInstance(w_team_rank,
                                                                 l_team_rank,
                                                                 CreatePreGameStats.get_field_goal_percent(eval_games, game.wteam),
                                                                 CreatePreGameStats.get_field_goal_percent(eval_games, game.lteam),
                                                                 CreatePreGameStats.get_free_throw_percent(eval_games, game.wteam),
                                                                 CreatePreGameStats.get_free_throw_percent(eval_games, game.lteam),
                                                                 CreatePreGameStats.get_orpg(eval_games, game.wteam),
                                                                 CreatePreGameStats.get_orpg(eval_games, game.lteam),
                                                                 CreatePreGameStats.get_drpg(eval_games, game.wteam),
                                                                 CreatePreGameStats.get_drpg(eval_games, game.lteam)))

                    # print len(training_set)

    @staticmethod
    def get_field_goal_percent(eval_games, team):
        fga = sum(eval_games[eval_games.wteam == team].wfga) + sum(eval_games[eval_games.lteam == team].lfga)
        fgm = sum(eval_games[eval_games.wteam == team].wfgm) + sum(eval_games[eval_games.lteam == team].lfgm)
        return float(fgm)/float(fga)

    @staticmethod
    def get_free_throw_percent(eval_games, team):
        den = sum(eval_games[eval_games.wteam == team].wfta) + sum(eval_games[eval_games.lteam == team].lfta)
        num = sum(eval_games[eval_games.wteam == team].wftm) + sum(eval_games[eval_games.lteam == team].lftm)
        return float(num)/float(den)

    @staticmethod
    def get_orpg(eval_games, team):
        den = sum(eval_games.wteam == team) + sum(eval_games.lteam == team)
        num = sum(eval_games[eval_games.wteam == team].wor) + sum(eval_games[eval_games.lteam == team].lor)
        return float(num)/float(den)

    @staticmethod
    def get_drpg(eval_games, team):
        den = sum(eval_games.wteam == team) + sum(eval_games.lteam == team)
        num = sum(eval_games[eval_games.wteam == team].wdr) + sum(eval_games[eval_games.lteam == team].ldr)
        return float(num)/float(den)

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
    def __init__(self, w_team_pts_rank, l_team_pts_rank, w_fgp, l_fgp, w_ftp, l_ftp, w_orpg, l_orpg, w_drpg, l_drpg):
        self.w_team_pts_rank = w_team_pts_rank
        self.l_team_pts_rank = l_team_pts_rank
        self.w_fgp = w_fgp
        self.l_fgp = l_fgp
        self.w_ftp = w_ftp
        self.l_ftp = l_ftp
        self.w_orpg = w_orpg
        self.l_orpg = l_orpg
        self.w_drpg = w_drpg
        self.l_drpg = l_drpg