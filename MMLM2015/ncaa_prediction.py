__author__ = 'Steve'
import pandas as ps
import create_ncaa_training_set as cn
import numpy as np
from math import log10, log

tourn_seeds_csv = 'data/tourney_seeds_2015.csv'
# tourn_seeds_csv = 'data/tourney_seeds.csv'

class PredictNCAASeason:
    def __init__(self):
        t_teams = ps.io.parsers.read_csv(tourn_seeds_csv)

        # seasons = [2011, 2012, 2013, 2014]
        seasons = [2015]
        self.p_mtx = []
        self.teams = []
        self.seasons = seasons


        for season in seasons:
            teams = t_teams[t_teams.season == season]
            self.teams.append(teams)
            season_games = cn.SeasonGames(season)

            pts_ranking = cn.CreateWeightedPtsRanking(season_games.season_games)

            vs_mtx = np.zeros((len(teams), len(teams)))

            for i, team1 in enumerate(teams.iterrows()):
                for j, team2 in enumerate(teams.iterrows()):
                    if team1[1].team != team2[1].team:
                        team1_idx = pts_ranking.teams.index(team1[1].team)
                        team2_idx = pts_ranking.teams.index(team2[1].team)

                        p = pts_ranking.team_rank[team1_idx]/pts_ranking.team_rank[team2_idx]
                        vs_mtx[i, j] = p

            self.p_mtx.append(vs_mtx)

    def CreatePredictionCSV(self):
        csv_strings = []

        f = open('test_submission6.csv', 'w')
        f.write('id,pred\n')

        for s, mtx in enumerate(self.p_mtx):

            for i, team1 in enumerate(self.teams[s].iterrows()):
                for j, team2 in enumerate(self.teams[s].iterrows()):
                    if j < i:
                        if team1[1].team < team2[1].team:
                            p = log10(mtx[i,j])*5.5 + 0.5
                            if p > 1.0:
                                p = 1.0
                            if p < 0.0:
                                p = 0.0
                            line_str = str(self.seasons[s]) + '_' + str(team1[1].team) + '_' + str(team2[1].team) + ',' + str(p) + '\n'
                        else:
                            p = log10(mtx[j, i])*5.5 + 0.5
                            if p > 1.0:
                                p = 1.0
                            if p < 0.0:
                                p = 0.0
                            line_str = str(self.seasons[s]) + '_' + str(team2[1].team) + '_' + str(team1[1].team) + ',' + str(p) + '\n'
                        f.write(line_str)

        f.close()


class ScoreSub:
    def __init__(self, test_file):
        sub = ps.io.parsers.read_csv(test_file)
        res = ps.io.parsers.read_csv('data/tourney_compact_results.csv')

        num_games = 0
        ll = 0.0

        eval_seasons = [2011, 2012, 2013, 2014]

        for season in eval_seasons:
            games = res[res.season == season]

            for game in games.iterrows():
                e_game = game[1]

                if e_game.wteam < e_game.lteam:
                    team1 = e_game.wteam
                    team2 = e_game.lteam
                    y = 1
                else:
                    team1 = e_game.lteam
                    team2 = e_game.wteam
                    y = 0

                id = str(season) + '_' + str(team1) + '_' + str(team2)
                p_res = sub[sub.id == id]
                p = p_res.irow(0).pred

                ll += y*log(p + .0001) + (1 - y)*log(1 - p+ .0001)
                num_games += 1

        self.ll = -ll/num_games

