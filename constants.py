import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

teammap = {
    'ARI':['Arizona Cardinals', 'Phoenix Cardinals', 'St. Louis Cardinals'],
    'ATL':['Atlanta Falcons'],
    'BAL':['Baltimore Ravens'],
    'BUF':['Buffalo Bills'],
    'CAR':['Carolina Panthers'],
    'CHI':['Chicago Bears'],
    'CIN':['Cincinnati Bengals'],
    'CLE':['Cleveland Browns'],
    'DAL':['Dallas Cowboys'],
    'DEN':['Denver Broncos'],
    'DET':['Detroit Lions'],
    'GB':['Green Bay Packers'],
    'HOU':['Houston Texans'],
    'IND':['Indianapolis Colts', 'Baltimore Colts'],
    'JAX':['Jacksonville Jaguars'],
    'KC':['Kansas City Chiefs'],
    'LAC':['Los Angeles Chargers', 'San Diego Chargers'],
    'LAR':['Los Angeles Rams', 'St. Louis Rams'],
    'LVR':['Las Vegas Raiders', 'Oakland Raiders', 'Los Angeles Raiders'],
    'MIA':['Miami Dolphins'],
    'MIN':['Minnesota Vikings'],
    'NE':['New England Patriots'],
    'NO':['New Orleans Saints'],
    'NYG':['New York Giants'],
    'NYJ':['New York Jets'],
    'PHI':['Philadelphia Eagles'],
    'PICK':['PICK'],
    'PIT':['Pittsburgh Steelers'],
    'SEA':['Seattle Seahawks'],
    'SF':['San Francisco 49ers'],
    'TB':['Tampa Bay Buccaneers'],
    'TEN':['Tennessee Titans', 'Tennessee Oilers', 'Houston Oilers'],
    'WAS':['Washington Commanders', 'Washington Football Team', 'Washington Redskins']
}