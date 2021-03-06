__author__ = 'Steve'
import numpy as np
import matplotlib.pylab as pl

class point_matrix:
    def __init__(self, season_data, teams):
        self.team_vect = []
        self.team_rank = []
        # find all teams
        for game in season_data.games:
            # does the winning team not exist?
            if game.w_team not in self.team_vect:
                self.team_vect.append(game.w_team)

            # does the losing team not exist?
            if game.l_team not in self.team_vect:
                self.team_vect.append(game.l_team)


        # create point mtx
        self.point_mtx = np.zeros((len(self.team_vect), len(self.team_vect)))
        self.numb_games = np.zeros(len(self.team_vect))
        self.numb_wins = np.zeros(len(self.team_vect))
        self.numb_loss = np.zeros(len(self.team_vect))
        for game in season_data.games:
            # get idx's
            w_idx = self.team_vect.index(game.w_team)
            l_idx = self.team_vect.index(game.l_team)

            # add game to count
            self.numb_games[w_idx] += 1
            self.numb_games[l_idx] += 1
            self.numb_wins[w_idx] += 1
            self.numb_loss[l_idx] += 1

            # add points to mtx
            self.point_mtx[w_idx, l_idx] += float(game.w_pts)/(float(game.w_pts + game.l_pts))
            self.point_mtx[l_idx, w_idx] += float(game.l_pts)/float((game.w_pts + game.l_pts))

        # remove teams
        # for idx in point_mtx_scale.transpose().sum(axis=1) == 0:
        #     print idx

        # scale matrix by num games
        D = np.diag(1.0/self.numb_games)
        point_mtx_scale = np.dot(D, self.point_mtx)

        V = np.linalg.eig(point_mtx_scale)[1]

        team_rank = abs(np.asarray(V[:, 0].real).transpose())
        # team_rank = team_rank[0,:]
        sort_idx = team_rank.argsort()
        sort_idx = sort_idx[::-1]
        #self.team_rank = self.team_vect[sort_idx]

        for idx in sort_idx:
            self.team_rank.append(self.team_vect[idx])

        for idx in range(0,100):
            print teams.team_name[teams.team_number.index(self.team_rank[idx])]

        # pl.plot(team_rank)
        # pl.show()






