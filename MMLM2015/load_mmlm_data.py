__author__ = 'Steve'

import csv

# define file locations
season_comp_filename = 'regular_season_compact_results.csv'
team_filename = 'teams.csv'
# season_detail_filename = 'regular_season_detailed_results.csv'


class load_data:
    def __init__(self):
        # read season file
        season_file = open(season_comp_filename, 'rt')
        season_csv = csv.reader(season_file, delimiter=',')

        self.seasons = []
        self.teams = cbb_teams()

        # load season data
        firstrow = True
        current_season = 0
        for row in season_csv:
            if firstrow:
                firstrow = False
            else:
                if int(row[0]) != current_season:
                    self.seasons.append(season_stats(int(row[0])))
                    current_season = int(row[0])

                self.seasons[-1].add_game(cbb_game(int(row[2]), int(row[4]), int(row[3]), int(row[5]), row[6]))
        season_file.close()

        # load team data
        team_file = open(team_filename, 'rt')
        team_csv = csv.reader(team_file, delimiter=',')
        firstrow = True
        for row in team_csv:
            if firstrow:
                firstrow = False
            else:
                self.teams.add_team(int(row[0]), row[1])
        team_file.close()


class season_stats:
    def __init__(self, year):
        self.year = year
        self.games = []

    def add_game(self, game):
        self.games.append(game)


class cbb_game:
    def __init__(self, w_team, l_team, w_pts, l_pts, w_loc):
        self.w_team = w_team
        self.l_team = l_team
        self.w_pts = w_pts
        self.l_pts = l_pts
        self.w_loc = w_loc


class cbb_teams:
    def __init__(self):
        self.team_number = []
        self.team_name = []

    def add_team(self, team_numb, team_name):
        self.team_number.append(team_numb)
        self.team_name.append(team_name)



