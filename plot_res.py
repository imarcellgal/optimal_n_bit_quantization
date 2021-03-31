"""Python code for creating interactive dashboard for visualizing
optimal reproduction points for n-bit quantization. To create visualizations.
To use it make sure to have the original results or generate new ones with the
get_reproduction.py script
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import pickle
from pdb import set_trace
import scipy.stats as st
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


with open("results/config.txt", "rb") as fp:   # Unpickling
    config = pickle.load(fp)
with open("results/res.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

orig_xhats = {}
for k,v in zip(b.keys(),b.values()): orig_xhats[k]=v


all_res = pd.read_csv("results/res.csv")

app.layout = html.Div(children=[
    html.H1(children='Optimal reproduction points for n-bit quantization'),
    html.Div([
    html.Div(
    children=['Slider for sigma',
    dcc.Slider(
    id= 'sigma-slider',
    min=1,
    max=all_res['sigma'].max(),
    value=1,
    included=False,
    marks={int(all_res['sigma'].min()): str(all_res['sigma'].min()), int(all_res['sigma'].max()): str(all_res['sigma'].max())},

),
html.Div(id='sigma-slider-output', style={'margin-top': 20}),
],style={'margin':'10px'}
),
html.Div(
children=['Slider for R',
    dcc.Slider(
    id= 'R-slider',
    min=all_res['R'].min(),
    max=all_res['R'].max(),
    value=all_res['R'].min(),
    included=False,
    marks={int(all_res['R'].min()): str(all_res['R'].min()), int(all_res['R'].max()): str(all_res['R'].max())},

),
html.Div(id='R-slider-output', style={'margin-top': 20}),
],style={'margin':'10px'}
)

],style={"display": "grid", "grid-template-columns": "50% 50%"}),

    html.Div(children='''
        The first plot shows the distortion and minimum distortion depending on
        R and sigma.
    ''',style={'margin':'10px'}),

    dcc.Graph(
        id='fig1',
    ),
    html.Div(children='''
        The second plot shows the optimal reproducion points for given sigma
        and R values
    '''),

    dcc.Graph(
        id='fig2',
    ),

])


@app.callback(
    dash.dependencies.Output('sigma-slider-output', 'children'),
    [dash.dependencies.Input('sigma-slider', 'value')])
def update_output(value):
    return 'Selected sigma value: {}'.format(value)
@app.callback(
    dash.dependencies.Output('R-slider-output', 'children'),
    [dash.dependencies.Input('R-slider', 'value')])
def update_output(value):
    return 'Selected R value: {}'.format(value)

@app.callback(
    dash.dependencies.Output('fig1', 'figure'),
    [dash.dependencies.Input('sigma-slider', 'value')])
def update_fig_1(value):
    fig = make_subplots(rows =1, cols = 2, subplot_titles=('Distortion and minimum distortion vs R for N(0,{})'.format(value),
    'Distortion/Minimum distortion vs R for N(0,{})'.format(value)))
    fig.add_trace(go.Scatter(name = 'Distortion', x=all_res.loc[all_res.sigma==value]['R'],
     y=all_res.loc[all_res.sigma==value]['distortion'], mode='lines'), row =1, col =1
     )
    fig.add_trace(go.Scatter(name = 'Minimum distortion', x=all_res.loc[all_res.sigma==value]['R'],
     y=all_res.loc[all_res.sigma==value]['D'], mode='lines'), row =1, col =1
     )
    fig.add_trace(go.Scatter(name = 'Distortion/Minimum distortion', x=all_res.loc[all_res.sigma==value]['R'],
     y=all_res.loc[all_res.sigma==value]['distortion']/all_res.loc[all_res.sigma==value]['D'], mode='lines'), row =1, col =2
     )
    fig['layout']['xaxis']['title']='R'
    fig['layout']['xaxis2']['title']='R'
    return fig

@app.callback(
    dash.dependencies.Output('fig2', 'figure'),
    [dash.dependencies.Input('sigma-slider', 'value'),
    dash.dependencies.Input('R-slider', 'value')])
def update_fig_1(sigma, R):
    data = orig_xhats[(R,sigma)]
    xhats = data[0]
    distortions = data[1]
    v_bounds = data[2]

    df = pd.DataFrame({'xrange':np.arange(-5*sigma,5*sigma,0.0025*sigma)})
    df['norm_x'] = st.norm(loc=0, scale=sigma).pdf(df['xrange'])

    fig = make_subplots(rows =1, cols =2,subplot_titles=('Xhat points for N(0,{})'.format(sigma),
      'Distortions for Voronoi regions'))

    fig.add_trace(go.Scatter(name = 'N(0,{})'.format(sigma), x=df['xrange'],
     y=df['norm_x'], mode='lines'),row=1,col=1
     )
    for i in range(len(xhats)):
        fig.add_trace(go.Scatter(name = 'xhat {}'.format(i), x=[xhats[i],xhats[i]],
         y=[0,df['norm_x'].max()], mode = 'lines'),row=1,col=1
         )
    for i in range(len(v_bounds)):
        fig.add_trace(go.Scatter(name = 'Voronoi region bound {}'.format(i), x=[v_bounds[i],v_bounds[i]],
         y=[0,df['norm_x'].max()], line=dict(dash='dot')),row=1,col=1
         )
    fig.add_trace(go.Bar(name = 'Distortions', x= ['Region {}'.format(i+1) for i in range(len(distortions))],
     y=data[1]),row=1,col=2
     )
    fig['layout']['yaxis']['title']='Probability density'.format(sigma)
    fig['layout']['xaxis']['title']='x'
    fig['layout']['xaxis2']['title']='Voronoi regions'
    fig['layout']['yaxis2']['title']='Distortion'
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
