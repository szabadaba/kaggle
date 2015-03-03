__author__ = 'Steve'

import csv
import pandas as ps

# define file locations
team_filename = 'data/teams.csv'
season_detail_filename = 'data/regular_season_detailed_results.csv'


class LoadSeasonData:
    def __init__(self):

        self.seasons = []
        self.teams = NCAATeams()

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

        # load seson data
        all_data = ps.io.parsers.read_csv(season_detail_filename)

        for season in all_data.season.unique():
            self.seasons.append(all_data[all_data.season == season])






class NCAATeams:
    def __init__(self):
        self.team_number = []
        self.team_name = []

    def add_team(self, team_numb, team_name):
        self.team_number.append(team_numb)
        self.team_name.append(team_name)