import numpy as np
from sklearn.preprocessing import LabelEncoder
from constants import teammap

# TODO: bigger dataset
def load_fivethirtyeight():
    pass

def load_spreadspoke():
    # load file
    fname = 'data/spreadspoke_scores.csv'
    # start from the beginning of the 1979-1980 season (row 2503)
    M = np.genfromtxt(fname, skip_header=2502, dtype=object, delimiter=',')
    # impute 
    M = impute_values(M)
    # add features
    M = add_features(M)
    # preprocess columns to build target
    M[:, 9] = M[:, 9].astype(np.float32) # spread
    M[:, 5] = M[:, 5].astype(int) # team0 score
    M[:, 6] = M[:, 6].astype(int) # team1 score
    Y = build_target(M)
    # preprocess other columns
    encoder = LabelEncoder()
    M[:, 0] = encoder.fit_transform(M[:, 0]) # date
    M[:, 1] = M[:, 1].astype(int) # season
    M[:, 2] = encoder.fit_transform(M[:, 2]) # week
    M[:, 3] = M[:, 3] == b'TRUE' # playoffs?
    M[:, 4] = encoder.fit_transform(M[:, 4]) # team0 team
    M[:, 7] = encoder.fit_transform(M[:, 7]) # team1 team
    M[:, 8] = encoder.fit_transform(M[:, 8]) # favorite team id
    M[:, 10] = M[:, 10].astype(np.float32) # o/u line
    M[:, 11] = encoder.fit_transform(M[:, 11]) # stadium
    M[:, 12] = M[:, 12] == b'TRUE' # neutral site?
    M[:, 13] = M[:, 13].astype(int) # weather temperature
    M[:, 14] = M[:, 14].astype(int) # weather wind
    M[:, 15] = M[:, 15].astype(int) # weather humidity
    M[:, 16] = encoder.fit_transform(M[:, 16]) # weather detail
    # get standardized X
    X = M[:, np.r_[0:5, 7:16]].astype(np.float32)
    mean_x = np.mean(X, axis=0)
    std_x = np.std(X, axis=0)
    X -= mean_x
    X /= std_x
    # train test split
    Xtrn, Xtst = X[:int(len(X) * 0.8), :], X[int(len(X) * 0.8):, :]
    Ytrn, Ytst = Y[:int(len(Y) * 0.8)], Y[int(len(Y) * 0.8):]
    # init models
    return Xtrn, Xtst, Ytrn, Ytst

def impute_values(M):
    tots = M[:, 10][~np.array([x is None or len(x) < 2 for x in M[:, 10]])]
    temps = M[:, 13][~np.array([x is None or len(x) == 0 for x in M[:, 13]])]
    winds = M[:, 14][~np.array([x is None or len(x) == 0 for x in M[:, 14]])]
    humids = M[:, 15][~np.array([x is None or len(x) == 0 for x in M[:, 15]])]
    median_total = np.nanmedian(tots.astype(float))
    median_temp = np.nanmedian(temps.astype(int))
    median_wind = np.nanmedian(winds.astype(int))
    median_humid = np.nanmedian(humids.astype(int))
    for i in range(len(M)):
        if len(M[i, 10]) == 1:
            M[i, 10] = median_total
        if len(M[i, 13]) == 0:
            M[i, 13] = median_temp
        if len(M[i, 14]) == 0:
            M[i, 14] = median_wind
        if len(M[i, 15]) == 0:
            M[i, 15] = median_humid
    return M

def build_target(M):
    diffs = M[:, 5] - M[:, 6]
    Y = [-1] * len(M)
    for i in range(len(M)):
        favorite_id = M[i, 8].decode()
        team0_name = M[i, 4].decode()
        team1_name = M[i, 7].decode()
        teams = teammap[favorite_id]
        spread = M[i, 9] # negative
        if team0_name in teams: # team0 fav
            if diffs[i] > -spread: # team0 covers
                Y[i] = 0
            elif diffs[i] < -spread: # team1 covers
                Y[i] = 1
            else: # push
                Y[i] = 2
        elif team1_name in teams: # team1 fav
            if diffs[i] < spread: # team1 covers
                Y[i] = 1
            elif diffs[i] > spread: # team0 covers
                Y[i] = 0
            else: # push
                Y[i] = 2
        else: # pick 'em (spread = 0)
            if diffs[i] > spread: # team0 covers
                Y[i] = 0
            elif diffs[i] < spread: # team1 covers
                Y[i] = 1
            else: # push
                Y[i] = 2
    return np.array(Y).astype(int)

def add_features(M):
    team0_favorite = [-1] * len(M)
    for i in range(len(M)):
        favorite_id = M[i, 8].decode()
        team0_name = M[i, 4].decode()
        team1_name = M[i, 7].decode()
        teams = teammap[favorite_id]
        team0_favorite[i] = 1 if team0_name in teams else -1 if team1_name in teams else 0
    M = np.column_stack((M, team0_favorite))
    return M
    # TODO: add features (record, points for/against, ATS record, O/U record) per season? 1y, 2y, 5y, 10y?
    # per team: W/L/D, points for/against, 
    seasons = np.unique(M[:, 1]).astype(int) # 1978, 1979, ..., 2022
    weeks = np.unique(M[:, 2]).astype(str)
    teams = list(np.unique(M[:, 4]).astype(str))

    encoding = {}
    for i in range(1, 19):
        encoding[i] = str(i)
    encoding[19] = 'Wildcard'
    encoding[20] = 'Division'
    encoding[21] = 'Conferece'
    encoding[22] = 'Superbowl'
    weeks[weeks == 'Wildcard'] = '19'
    weeks[weeks == 'Division'] = '20'
    weeks[weeks == 'Conference'] = '21'
    weeks[weeks == 'Superbowl'] = '22'
    weeks = np.sort(weeks.astype(int)) # 1, 2, 3, ..., 18, Wildcard, Division, Conference, Super Bowl (per season)

    # team0_win, team0_loss, team0_draw, team0_pf, team0_pa,
    # team0_spread_win, team0_spread_loss, team0_spread_push,
    # team0_ou_win, team0_ou_loss, team0_ou_draw,
    # and repeat all above for team1

    # more betting features:
    # favorite_spread_win, favorite_spread_loss, favorite_spread_push
    # dog_spread_win, dog_spread_loss, dog_spread_push
    # fav_ou_win, fav_ou_loss, fav_ou_push
    # dog_ou_win, dog_ou_loss, dog_ou_push

    # combine features:
    # fav_team0_spread_win, fav_team1_spread_win, fav_neutral_spread_win, ...
    # regular_season_spread_win, playoff_spread_win, week1_spread_win, ...

    # TODO: normalize weeks
    
    num_new_features = 11 * 2
    features = np.zeros((len(M), num_new_features))
    for season in seasons:
        for week in weeks:
            week_games = M[(M[:, 1] == str(season).encode()) & (M[:, 2] == encoding[week].encode())]
            for j in range(len(week_games)):
                team0_index, team1_index = teams.index(week_games[j][4].decode()), teams.index(week_games[j][7].decode())
                spread = -1 * float(week_games[j][9].decode())
                print(spread)

    M = np.concatenate([M, features], axis=1)

    return M