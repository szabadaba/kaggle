__author__ = 'Steven Szabados'

# from joblib import Parallel, delayed
# import multiprocessing
import pandas as ps
import numpy as np
from math import sqrt
import sys


season_detail_filename = 'data/regular_season_detailed_results_2015.csv'
# season_detail_filename = 'data/regular_season_detailed_results.csv'

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
                                        te.w_drpg/te.l_drpg,
                                        te.w_topg/te.l_topg,
                                        te.w_3pp/te.l_3pp,
                                        te.w_home])
            # self.formatted_data.append([te.w_team_pts_rank,
            #                             te.w_fgp,
            #                             te.w_ftp,
            #                             te.w_orpg,
            #                             te.w_drpg,
            #                             te.w_topg,
            #                             te.w_3pp])
            self.truth_label.append(1)
            self.formatted_data.append([te.l_team_pts_rank/te.w_team_pts_rank,
                                        te.l_fgp/te.w_fgp,
                                        te.l_ftp/te.w_ftp,
                                        te.l_orpg/te.w_orpg,
                                        te.l_drpg/te.w_drpg,
                                        te.l_topg/te.w_topg,
                                        te.l_3pp/te.w_3pp,
                                        te.l_home])
            # self.formatted_data.append([te.l_team_pts_rank,
            #                             te.l_fgp,
            #                             te.l_ftp,
            #                             te.l_orpg,
            #                             te.l_drpg,
            #                             te.l_topg,
            #                             te.l_3pp])
            self.truth_label.append(0)


class SeasonGames:
    def __init__(self, season):
        data = ps.io.parsers.read_csv(season_detail_filename)
        self.season_games = data[data.season == season]


class CreateTestSet:
    def __init__(self, season, team1, team2):
        # get season data
        data = ps.io.parsers.read_csv(season_detail_filename)
        season_data = data[data.season == season]

        #use pts ranking
        pts_rank = CreateWeightedPtsRanking(season_data)

        # set pts rank
        team1_pts_rank = pts_rank.team_rank[pts_rank.teams.index(team1)]
        team2_pts_rank = pts_rank.team_rank[pts_rank.teams.index(team2)]

        team1_wgames = season_data[season_data.wteam == team1]
        team1_lgames = season_data[season_data.lteam == team1]
        team2_wgames = season_data[season_data.wteam == team2]
        team2_lgames = season_data[season_data.lteam == team2]

        # create training instance
        self.training_set = NCAAGameTrainingInstance(team1_pts_rank,
                                                 team2_pts_rank,
                                                 CreatePreGameStats.get_field_goal_percent(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_field_goal_percent(team2_wgames,team2_lgames),
                                                 CreatePreGameStats.get_free_throw_percent(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_free_throw_percent(team2_wgames, team2_lgames),
                                                 CreatePreGameStats.get_orpg(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_orpg(team2_wgames, team2_lgames),
                                                 CreatePreGameStats.get_drpg(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_drpg(team2_wgames, team2_lgames),
                                                 CreatePreGameStats.get_topg(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_topg(team2_wgames, team2_lgames),
                                                 CreatePreGameStats.get_3p_percent(team1_wgames, team1_lgames),
                                                 CreatePreGameStats.get_3p_percent(team2_wgames, team2_lgames))



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

                    w_team_wgames = eval_games[eval_games.wteam == game.wteam]
                    w_team_lgames = eval_games[eval_games.lteam == game.wteam]
                    l_team_wgames = eval_games[eval_games.wteam == game.lteam]
                    l_team_lgames = eval_games[eval_games.lteam == game.lteam]

                    # create training instance
                    training_set.append(NCAAGameTrainingInstance(w_team_rank,
                                                                 l_team_rank,
                                                                 CreatePreGameStats.get_field_goal_percent(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_field_goal_percent(l_team_wgames,l_team_lgames),
                                                                 CreatePreGameStats.get_free_throw_percent(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_free_throw_percent(l_team_wgames, l_team_lgames),
                                                                 CreatePreGameStats.get_orpg(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_orpg(l_team_wgames, l_team_lgames),
                                                                 CreatePreGameStats.get_drpg(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_drpg(l_team_wgames, l_team_lgames),
                                                                 CreatePreGameStats.get_topg(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_topg(l_team_wgames, l_team_lgames),
                                                                 CreatePreGameStats.get_3p_percent(w_team_wgames, w_team_lgames),
                                                                 CreatePreGameStats.get_3p_percent(l_team_wgames, l_team_lgames),
                                                                 game.wloc == 'H',
                                                                 game.wloc == 'A'))

                    # print len(training_set)

    @staticmethod
    def get_field_goal_percent(w_games, l_games):
        fga = sum(w_games.wfga) + sum(l_games.lfga)
        fgm = sum(w_games.wfgm) + sum(l_games.lfgm)
        return float(fgm)/float(fga)
    @staticmethod
    def get_3p_percent(w_games, l_games):
        den = sum(w_games.wfga3) + sum(l_games.lfga3)
        num = sum(w_games.wfgm3) + sum(l_games.lfgm3)
        return float(num)/float(den)

    @staticmethod
    def get_free_throw_percent(w_games, l_games):
        den = sum(w_games.wfta) + sum(l_games.lfta)
        num = sum(w_games.wftm) + sum(l_games.lftm)
        return float(num)/float(den)

    @staticmethod
    def get_orpg(w_games, l_games):
        den = len(w_games.wteam) + len(l_games.lteam)
        num = sum(w_games.wor) + sum(l_games.lor)
        return float(num)/float(den)

    @staticmethod
    def get_drpg(w_games, l_games):
        den = len(w_games.wteam) + len(l_games.lteam)
        num = sum(w_games.wdr) + sum(l_games.ldr)
        return float(num)/float(den)

    @staticmethod
    def get_topg(w_games, l_games):
        den = len(w_games.wteam) + len(l_games.lteam)
        num = sum(w_games.wto) + sum(l_games.lto)
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

            # if row[1].wloc == 'H':
            #     dampen_factor = 0.75
            # elif row[1].wloc == 'A':
            #     dampen_factor = 1.25


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
    def __init__(self, w_team_pts_rank, l_team_pts_rank, w_fgp, l_fgp, w_ftp, l_ftp, w_orpg, l_orpg, w_drpg, l_drpg, w_topg, l_topg, w_3pp, l_3pp, w_home, l_home):
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
        self.w_topg = w_topg
        self.l_topg = l_topg
        self.w_3pp = w_3pp
        self.l_3pp = l_3pp
        self.w_home = w_home
        self.l_home = l_home