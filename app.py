import numpy as np
import pandas as pd
from enum import Enum, auto
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots

MSFT_PATH = "https://stooq.pl/q/d/l/?s=msft.us&d1=20160301&d2=20170301&i=d"  # kurs MSFT 01-03-2016 do 01-03-2017
GOOG_PATH = "https://stooq.pl/q/d/l/?s=goog.us&d1=20160301&d2=20170301&i=d"  # kurs GOOG 01-03-2016 do 01-03-2017
TAU_MAX = "\u03C4_MAX"
TAU_MIN = "\u03C4_MIN"

CONST_R = 0.03 / 250


def sm(p, returns):  # estymacja s i m przy ustalonym p
    e = np.mean(returns)
    v = np.mean(returns ** 2) - e ** 2
    m = np.sqrt(v / (4 * p * (1 - p)))
    s = e - m * (2 * p - 1)
    return s, m


def p_est(p, returns):  # estymacja p przy użyciu (*)
    (s, m) = sm(p, returns)
    return np.count_nonzero(np.abs(returns - (s + m)) < np.abs(returns - (s - m))) / len(returns)


def p_opt(returns, stock_name, N=10000):
    p = np.array(range(1, N)) / N
    p_fit = np.zeros(N - 1)
    for i in range(0, N - 1):
        p_fit[i] = np.abs(p[i] - p_est(p[i], returns))
    plt.plot(p, p_fit)
    plt.xlabel("X axis label")
    plt.ylabel("|p - p_estimated|")
    plt.title(f"estimate of p for {stock_name}")
    return p[np.where(p_fit == np.amin(p_fit))]


class ModelParams():
    r: float

    def __init__(self, p, s, m, q, r=CONST_R):
        self.p = p
        self.s = s
        self.m = m
        self.q = q
        self.r = r

    def __str__(self):
        return f"p = {self.p.round(decimals=5)}\ns = {self.s.round(decimals=5)}\nm = {self.m.round(decimals=5)}\nq = {self.q.round(decimals=5)}"


def read_data(path):
    df = pd.read_csv(path)
    df['Data'] = pd.to_datetime(df['Data'])
    return df


def GetMartingaleMeasure(s, m, r=CONST_R):  # returns q
    q = (np.exp(r - s) - np.exp(-1 * m)) / (np.exp(m) - np.exp(-m))
    assert q > 0
    assert q < 1
    return q


def estimate_params(df, stock_name):
    logreturns = np.log1p(df.Otwarcie.pct_change())[1:].to_numpy()  # zwroty logarytmiczne, ceny z otwarcia
    p = p_opt(logreturns, stock_name)
    (s, m) = sm(p, logreturns)
    p, m, s = p[0], m[0], s[0]
    q = GetMartingaleMeasure(s, m)
    return ModelParams(p, s, m, q)


mstf_dividends = {
    31: 0.06,
    93: 0.06,
    155: 0.06,
    217: 0.06
}
msft = read_data(MSFT_PATH)
msft_params = estimate_params(msft, "MSFT")
goog = read_data(GOOG_PATH)
goog_params = estimate_params(goog, "GOOG")

TRAIN_START = np.min(msft['Data'])
TRAIN_END = np.max(msft['Data'])


class OptionType(Enum):
    AMERICAN_CALL = auto()
    AMERICAN_PUT = auto()


class Option:
    def __init__(self, type, days_till_expiry, threshold, under_asset_name):
        """
        Args:
          - type one of OptionType
          - days_till_expiry
          - threshold
        """
        self.type = type
        self.days_till_expiry = days_till_expiry
        self.threshold = threshold
        self.under_asset_name = under_asset_name
        if self.type == OptionType.AMERICAN_CALL:
            self.payoff = lambda x: max(x - self.threshold, 0)
        elif self.type == OptionType.AMERICAN_PUT:
            self.payoff = lambda x: max(self.threshold - x, 0)
        else:
            raise Exception(f"Unknown option type: {self.type}.")

    def Payoff(self, s):
        return self.payoff(s)

    def __str__(self):
        return f"{self.type.name} on {self.under_asset_name} @ {self.threshold}, expiry in {self.days_till_expiry} days."


def _on_board(i, j):
    return i >= 0 and j >= 0 and j <= i


def get_info_v2(i, j, U, H, K, reachable_with_0_K, get_reachable_with_U_over_H):
    if reachable_with_0_K[i][j] and K[i][j] > 0:
        return TAU_MAX
    if get_reachable_with_U_over_H[i][j] and U[i][j] == H[i][j]:
        return TAU_MIN
    return None


def get_reachable_with_U_over_H(U, H, N):
    reachable_with_U_over_H = np.zeros((N, N), dtype=bool)
    reachable_with_U_over_H[0][0] = True

    for i in range(1, N):
        for j in range(0, i + 1):
            above_parent = _on_board(i - 1, j)
            below_parent = _on_board(i - 1, j - 1)
            if above_parent and reachable_with_U_over_H[i - 1][j] and U[i - 1][j] > H[i - 1][j]:
                reachable_with_U_over_H[i][j] = True
            elif below_parent and reachable_with_U_over_H[i - 1][j - 1] and U[i - 1][j - 1] > H[i - 1][j - 1]:
                reachable_with_U_over_H[i][j] = True

    return reachable_with_U_over_H


def get_reachable_with_0_K(K, N):
    reachable_with_0_K = np.zeros((N - 1, N - 1), dtype=bool)
    reachable_with_0_K[0][0] = True

    for i in range(1, N - 1):
        for j in range(0, i + 1):
            above_parent = _on_board(i - 1, j)
            below_parent = _on_board(i - 1, j - 1)
            if above_parent and reachable_with_0_K[i - 1][j] and K[i - 1][j] == 0:
                reachable_with_0_K[i][j] = True
            elif below_parent and reachable_with_0_K[i - 1][j - 1] and K[i - 1][j - 1] == 0:
                reachable_with_0_K[i][j] = True

    return reachable_with_0_K


def u_eq_h_color(u, h):
    if u != h:
        return -10
    if u == h:
        return np.log1p(u)


def VanillaAmericanOptionStats(model_params, tdy_asset_price, option, dividends=dict()):
    """
    Args:
      - q - miara mtg, że w_i = 1
      - s, m, r - z kalibracji modelu
      - tdy_asset_price - dzisiejsza cena instrumentu podstawowego
      - option - opcja do wyceny
    Returns:
      - S[][] - drzewko cen aktywa bazowego: tablica trójkątna: S[n = numer_dnia][liczba w_i == 1 dla i \in {0, 1, ..., n} ].
      - U[][] - otoczka Snella; tablica trójkątna jak wyżej.
      - K[][] - przyrosty procesu przewidywalnego z rozkładu Dooba -  tablice trójkątna wymiaru (n-1) x (n-1)
      - Opt[][] - optymalne czasy wykonania - tablica True/ False; tablica trójkątna jak wyżej wymiaru (n-1) x (n-1)
      - days[][] - numery dnia dopowiadajace indeksom tj. days[i][j] = i
      - df - pandas.DataFrame days, S, U, K w 'long format' - na potrzeby wykresów
    """
    s, m, r, q = model_params.s, model_params.m, model_params.r, model_params.q
    N = option.days_till_expiry + 1
    S = np.zeros((N, N))
    S[0][0] = tdy_asset_price
    U = np.zeros_like(S)
    H = np.zeros_like(S)
    K = np.zeros((N - 1, N - 1))
    Opt = np.zeros_like(K, dtype=bool)
    df_rows = []
    for i in range(1, N):
        S[i][0] = S[i - 1][0] * np.exp(s - m)
        if i in dividends:
            S[i][0] *= (1. - dividends[i])

        for j in range(1, i + 1):
            S[i][j] = S[i - 1][j - 1] * np.exp(s + m)
            if i in dividends:
                S[i][j] *= (1. - dividends[i])
    # W liściach, U[N-1][_] to zdyskontowany payoff.
    for i in range(0, N):
        U[N - 1][i] = option.Payoff(S[N - 1][i]) * np.exp(-r * (N - 1))
        H[N - 1][i] = U[N - 1][i]
        u_eq_h = u_eq_h_color(U[N - 1][i], H[N - 1][i])
        df_rows.append({'day': N - 1, 'S': S[N - 1][i], 'U': U[N - 1][i], 'K': None, 'K_positive': None, 'U==H': u_eq_h,
                        'info': TAU_MAX})

    # Wyliczamy pozostałe wartości U[][].
    for i in range(N - 2, -1, -1):
        for j in range(0, i + 1):
            cond_exp = q * U[i + 1][j + 1] + (1. - q) * U[i + 1][j]
            early_exercise = option.Payoff(S[i][j]) * np.exp(-r * i)
            H[i][j] = early_exercise
            U[i][j] = max(cond_exp, early_exercise)
            if early_exercise >= cond_exp:
                Opt[i][j] = True
            K[i][j] = U[i][j] - cond_exp

    reachable_with_0_K = get_reachable_with_0_K(K, N)
    reachable_with_U_over_H = get_reachable_with_U_over_H(U, H, N)

    for i in range(N - 2, -1, -1):
        for j in range(0, i + 1):
            # info = _get_info(i-1, j-1, i, j, U, H, K)
            # if info is None:
            #   info = get_info(i-1, j, i, j, U, H, K)
            info = get_info_v2(i, j, U, H, K, reachable_with_0_K, reachable_with_U_over_H)
            u_eq_h = u_eq_h_color(U[i][j], H[i][j])
            df_rows.append(
                {'day': i, 'S': S[i][j], 'U': U[i][j], 'K': K[i][j], 'K_positive': K[i][j] > 0, 'U==H': u_eq_h,
                 'info': info})

    days = np.arange(N)[None, :]
    days = np.tile(days.T, (1, N))

    df = pd.DataFrame(df_rows)
    df['K_positive'] = df['K_positive'].astype(str)
    # df['U==H'] = df['U==H'].astype(str)
    df['info'] = df['info'].astype(str)

    df['S_log'] = np.log(df['S'])

    return S, U, K, Opt, days, df


def _set_opacity(info):
    return 0.2 if info == 'None' else 1.0


def _set_color(info):
    if info == TAU_MAX:
        return 'black'
    if info == TAU_MIN:
        return 'green'
    return 'orange'


def plot_opt_times(option, df):
    fig = go.Figure(data=[go.Scatter(
        x=df['day'],
        y=df['S'],
        mode='markers',
        marker=dict(
            color=list(map(_set_color, df['info'])),
            # color_discrete_map = {TAU_MAX: 'black', TAU_MIN: 'green', 'None':'orange'},
            opacity=list(map(_set_opacity, df['info'])),
            # showlegend = True
        )
    )])
    fig.update_layout(title=str(option), xaxis_title='day', yaxis_title='S')
    fig.update_yaxes(type='log')

    # tight layout
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    return fig


def get_option_params(option):
    df = msft
    model_params = msft_params
    dividends = mstf_dividends
    if option.under_asset_name == 'GOOG':
        df = goog
        model_params = goog_params
        dividends = dict()
    return df, model_params, dividends


NUM_TR_DAYS = 200
options = [Option(OptionType.AMERICAN_CALL, NUM_TR_DAYS, 900, 'GOOG'),
           Option(OptionType.AMERICAN_PUT, NUM_TR_DAYS, 900, 'GOOG'),
           Option(OptionType.AMERICAN_CALL, NUM_TR_DAYS, 60, 'MSFT'),
           Option(OptionType.AMERICAN_PUT, NUM_TR_DAYS, 60, 'MSFT')]
option_types = [OptionType.AMERICAN_CALL,
                OptionType.AMERICAN_PUT]


def get_hedging_portfolio(S, U, r, i, j):
    x = (U[i + 1][j + 1] - U[i + 1][j]) / (S[i + 1][j + 1] - S[i + 1][j])  # num of stocks
    y = (U[i + 1][j + 1] - x * S[i + 1][j + 1]) * np.exp(-r)  # num of numerer
    return x, y


def get_trajectory_data(trajectory, S, U, K, r=CONST_R):
    """
    trajectory should be a list of 0s and 1s. O represents that the asset moved down, 1 that up.
    """
    N = S.shape[0]
    Xt = np.zeros((N,))
    Xt[0] = S[0][0]
    Ut = np.zeros((N,))
    Ut[0] = U[0][0]
    j = 0
    for i in range(1, N):
        j += trajectory[i - 1]
        Xt[i] = S[i][j] * np.exp(-r * i)
        Ut[i] = U[i][j]

    At = np.zeros((N,))
    At[0] = 0
    At[1] = K[0][0]
    j = 0
    for i in range(1, N - 1):
        j += trajectory[i - 1]
        At[i + 1] = At[i] + K[i][j]

    Stockt = np.zeros((N - 1,))
    Casht = np.zeros((N - 1,))
    j = 0
    for i in range(0, N - 1):
        Stockt[i], Casht[i] = get_hedging_portfolio(S, U, r, i, j)
        j += trajectory[i]

    Vt = Stockt * Xt[:N - 1] + Casht

    return Xt, Ut, At, Stockt, Casht, Vt


def plot_hedging_strategy(option, trajectory):
    df, model_params, dividends = get_option_params(option)
    tdy_under_asset_price = df.iloc[-1].Otwarcie
    S, U, K, Opt, days, df = VanillaAmericanOptionStats(model_params, tdy_under_asset_price, option, dividends)
    N = K.shape[0]
    St, Ut, At, Xt, Yt, Vt = get_trajectory_data(trajectory, S, U, K)
    time = np.arange(1, N + 1)
    fig = make_subplots(rows=3, cols=1)
    fig.add_trace(go.Scatter(x=time, y=St, mode='lines', name='X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=At, mode='lines', name='A'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=Ut, mode='lines', name='U'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=Vt, mode='markers', name='V', opacity=0.5,
                             marker=dict(color='black', symbol='circle-open')), row=1, col=1)
    fig.add_trace(go.Scatter(x=time, y=Xt, mode='lines', name='num_stock'), row=2, col=1)
    fig.add_trace(go.Scatter(x=time, y=Yt, mode='lines', name='numerer'), row=3, col=1)
    fig.update_layout(height=1000, width=1500, title_text="Hedging strategy for " + str(option))
    return fig


def get_trajectory(desc="random", N=NUM_TR_DAYS + 1):
    if desc == "ascending":
        return np.ones(N, dtype=int)
    if desc == "descending":
        return np.zeros(N, dtype=int)
    if desc == "random":
        return np.random.randint(low=0, high=2, size=N, dtype=int)
    if desc == "ups-and-downs":
        return 1 - get_trajectory("downs-and-ups", N)
    if desc == "downs-and-ups":
        return np.hstack([np.zeros(N // 4, dtype=int), np.ones(N // 4, dtype=int), np.zeros(N // 4, dtype=int),
                          np.ones(N // 4, dtype=int), np.zeros(N % 4, dtype=int)])
    raise IndexError


N = NUM_TR_DAYS + 1

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.Div([html.H6("Optymalne czasy wykonania opcji")], style={'float': 'center'}),
        html.Div([
            html.Div([
                html.Div([

                    html.Label("Asset:"),
                    dcc.RadioItems(
                        id='asset',
                        options=[{'label': 'Google (GOOG.US)', 'value': 'GOOG'},
                                 {'label': 'Microsoft (MSFT.US)', 'value': 'MSFT'},
                                 ],
                        value='GOOG',
                        style={'marginRight': '40px'}
                    ),
                ]),

                html.Div([
                    html.Label("Option type:"),
                    dcc.RadioItems(
                        id='option_type',
                        options=[{'label': 'American Put', 'value': 1},
                                 {'label': 'American Call', 'value': 0},
                                 ],
                        value=0,
                        style={'marginRight': '10px'}
                    ),
                ]),
            ]),
            html.Div(["Initial value: ",
                      dcc.Input(id="initial_value", type="number", value=60.098, style={'marginRight': '20px'}, min=0.0,
                                max=3000.0,
                                step=.001,
                                size="7"), ], style={'float': 'right'}),
            html.Div(["Strike: ",
                      dcc.Input(id="strike", type="number", value=60, style={'marginRight': '20px'}, min=0.0,
                                max=3000.0, step=.001,
                                size="7"), ], style={'float': 'right'}),
            html.Div(["Risk-free rate: ",
                      dcc.Input(id="rate", type="number", value=0.03, style={'marginRight': '20px'}, min=0.0, max=.99,
                                step=.01,
                                size="7")], style={'float': 'right'}),
            html.Div(["m: ",
                      dcc.Input(id="m", type="number", value=0.01, style={'marginRight': '20px'}, min=0.0, max=1,
                                step=.000001,
                                size="7")], style={'float': 'right'}),
            html.Div(["s: ",
                      dcc.Input(id="s", type="number", value=0.0001, style={'marginRight': '20px'}, min=0.0, max=1,
                                step=.000001,
                                size="7"), ], style={'float': 'right'}),
            html.Div([html.Button(id='submit-button-state', n_clicks=0, children='Plot')],
                     style={'float': 'right', 'marginRight': '20px'}),
        ], style={'display': 'inline-block', 'width': '12%'}),
        html.Div([
            dcc.Graph(id='indicator-graphic1'),
            dcc.Graph(id='indicator-graphic2'),
            dcc.Graph(id='indicator-graphic3'),
        ], style={'display': 'inline-block', 'width': '87%', 'float': 'right'}),
    ]),

    html.Br(),
    html.Div([html.H6("Strategia zabezpieczająca na przykładowych trajektoriach:")],
             style={'float': 'center', 'marginTop': 1000}),
    html.Div([

        html.Div([
            html.Div([

                html.Label("Asset: "),
                dcc.RadioItems(
                    id='asset_trajectory',
                    options=[{'label': 'Google (GOOG.US)', 'value': 'GOOG'},
                             {'label': 'Microsoft (MSFT.US)', 'value': 'MSFT'},
                             ],
                    value='GOOG',
                    style={'marginRight': '40px'}
                ),
            ]),

            html.Div([
                html.Label("Option type: "),
                dcc.RadioItems(
                    id='option_type_trajectory',
                    options=[{'label': 'American Put', 'value': 1},
                             {'label': 'American Call', 'value': 0},
                             ],
                    value=0,
                    style={'marginRight': '10px'}
                ),
            ]),
            html.Label("Trajectory type:"),
            dcc.RadioItems(
                id='trajectory_type',
                options=[{'label': "ascending", 'value': "ascending"},
                         {'label': "descending", 'value': "descending"},
                         {'label': "random", 'value': "random"},
                         {'label': "ups and downs", 'value': "ups-and-downs"},
                         {'label': "downs and ups", 'value': "downs-and-ups"}
                         ],
                value="random",
                style={'marginRight': '10px'}
            ),
            html.Div([html.Button(id='button-trajectory', n_clicks=0, children='Plot')]),

        ], style={'display': 'inline-block', 'width': '12%'}),
        html.Div([
            dcc.Graph(id='trajectory')
        ], style={'display': 'inline-block', 'width': '87%', 'float': 'right'})
    ]),
])


@app.callback(
    Output('initial_value', 'value'),
    Output('strike', 'value'),
    Output('m', 'value'),
    Output('s', 'value'),
    Input('asset', 'value')
)
def update_value(asset):
    if asset == 'GOOG':
        model_params = estimate_params(goog, 'GOOG')
        ret = goog.iloc[-1].Otwarcie, 900, round(model_params.m, 6), round(model_params.s, 6)
    else:
        model_params = estimate_params(msft, 'MSFT')
        ret = msft.iloc[-1].Otwarcie, 60, round(model_params.m, 6), round(model_params.s, 6)
    return ret


@app.callback(
    Output('indicator-graphic1', 'figure'),
    Output('indicator-graphic2', 'figure'),
    Output('indicator-graphic3', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('option_type', 'value'),
    State('asset', 'value'),
    State('rate', 'value'),
    State('strike', 'value'),
    State('initial_value', 'value'),
    State('m', 'value'),
    State('s', 'value'),
)
def update_graph(n_clicks, option_type, asset, rate, strike, initial_value, m, s):
    option = Option(option_types[option_type], NUM_TR_DAYS, strike, asset)
    df, model_params, dividends = get_option_params(option)
    model_params.r = float(rate) / 250
    model_params.m = m
    model_params.s = s
    model_params.q = GetMartingaleMeasure(s, m, model_params.r)
    tdy_under_asset_price = initial_value
    S, U, K, Opt, days, df = VanillaAmericanOptionStats(model_params, tdy_under_asset_price, option, dividends)
    fig = (
        px.scatter(df, x='day', y='S', opacity=0.7, color='K', hover_data=['K', 'S'], title=str(option), log_y=True,
                   color_continuous_scale=px.colors.sequential.Jet),
        # px.scatter(df, x='day', y='S', opacity=0.7, color='U==H', color_discrete_map = {'True': 'darkorange',
        # 'False': 'green'}, hover_data=['K', 'S'], title=str(option), log_y=True),
        px.scatter(df, x='day', y='S', opacity=0.7, color='U==H', hover_data=['K', 'S', 'U'],
                   title=str(option) + ": log(U+1) if U == H else -10", log_y=True),
        plot_opt_times(option, df))
    return fig


@app.callback(
    Output("trajectory", 'figure'),
    Input('button-trajectory', 'n_clicks'),
    State('asset_trajectory', 'value'),
    State('option_type_trajectory', 'value'),
    State("trajectory_type", 'value')
)
def update_trajectory(n_clicks, asset, option_type, trajectory_type):
    if asset == 'GOOG' and option_type == 0:
        option = options[0]
    elif asset == 'GOOG' and option_type == 1:
        option = options[1]
    elif asset == 'MSFT' and option_type == 0:
        option = options[2]
    else:
        option = options[3]
    fig = plot_hedging_strategy(option, get_trajectory(trajectory_type))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
