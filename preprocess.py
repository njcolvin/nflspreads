import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load():
    # load file
    fname = 'data/spreadspoke_scores.csv'
    # start from the beginning of the 1979-1980 season (row 2503)
    M = np.genfromtxt(fname, skip_header=2502, dtype=object, delimiter=',')

    # impute 
    M = impute_values(M)

    # TODO: add features (record, points for/against, ATS record, O/U record) per season? 1y, 2y, 5y, 10y?
    # M = add_features(M)
    print(np.unique(M[:, 8]))

    # preprocess
    encoder = LabelEncoder()
    M[:, 0] = encoder.fit_transform(M[:, 0])
    M[:, 1] = M[:, 1].astype(int)
    M[:, 2] = encoder.fit_transform(M[:, 2])
    M[:, 3] = M[:, 3] == b'TRUE'
    M[:, 4] = encoder.fit_transform(M[:, 4])
    M[:, 5] = M[:, 5].astype(int)
    M[:, 6] = M[:, 6].astype(int)
    M[:, 7] = encoder.fit_transform(M[:, 7])
    M[:, 8] = encoder.fit_transform(M[:, 8])
    M[:, 9] = M[:, 9].astype(np.float32)
    M[:, 10] = M[:, 10].astype(np.float32)
    M[:, 11] = encoder.fit_transform(M[:, 11])
    M[:, 12] = M[:, 12] == b'TRUE'
    M[:, 13] = M[:, 13].astype(int)
    M[:, 14] = M[:, 14].astype(int)
    M[:, 15] = M[:, 15].astype(int)
    M[:, 16] = encoder.fit_transform(M[:, 16])


    X = M[:, np.r_[0:5, 7:17]].astype(np.float32)
    Y = M[:, 5:7].astype(np.float32)

    # standardize
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    std_x = np.std(X)
    std_y = np.std(Y)

    X -= mean_x
    X /= std_x
    #Y -= mean_y
    #Y /= std_y

    # split into training and test sets
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(X, Y)

    # init models
    mlp = MLPRegressor(solver='sgd', batch_size=100, activation='tanh').fit(Xtrn, Ytrn)
    dt = DecisionTreeRegressor().fit(Xtrn, Ytrn)
    lr = LinearRegression().fit(Xtrn, Ytrn)
    gd = MultiOutputRegressor(GradientBoostingRegressor()).fit(Xtrn, Ytrn)

    # evaluate models
    models = [mlp, dt, lr, gd]
    preds = [model.predict(Xtst) for model in models]
    evals = [(mean_absolute_error(Ytst, pred), mean_squared_error(Ytst, pred), r2_score(Ytst, pred)) for pred in preds]

    # print results
    for i in range(len(models)):
        print()
        print(type(models[i]).__name__)
        print("--------------------------------------")
        print('MAE is {}'.format(evals[i][0]))
        print('MSE is {}'.format(evals[i][1]))
        print('R2 score is {}'.format(evals[i][2]))
        print('actual: ' + str(Ytst[:5]))
        print('predicted: ' + str(preds[:5][0]))
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

def add_features(M):
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

    # home_win, home_loss, home_draw, home_pf, home_pa,
    # home_spread_win, home_spread_loss, home_spread_push,
    # home_ou_win, home_ou_loss, home_ou_draw,
    # and repeat all above for away

    # more betting features:
    # favorite_spread_win, favorite_spread_loss, favorite_spread_push
    # dog_spread_win, dog_spread_loss, dog_spread_push
    # fav_ou_win, fav_ou_loss, fav_ou_push
    # dog_ou_win, dog_ou_loss, dog_ou_push

    # combine features:
    # fav_home_spread_win, fav_away_spread_win, fav_neutral_spread_win, ...
    # regular_season_spread_win, playoff_spread_win, week1_spread_win, ...

    # TODO: normalize weeks
    
    num_new_features = 11 * 2
    features = np.zeros((len(M), num_new_features))
    for season in seasons:
        for week in weeks:
            week_games = M[(M[:, 1] == str(season).encode()) & (M[:, 2] == encoding[week].encode())]
            for j in range(len(week_games)):
                home_index, away_index = teams.index(week_games[j][4].decode()), teams.index(week_games[j][7].decode())
                spread = -1 * float(week_games[j][9].decode())
                print(spread)

    M = np.concatenate([M, features], axis=1)

    return M