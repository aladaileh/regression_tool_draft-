import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots
from datetime import datetime


def plot_xs_vs_ys(country, inputs_df, xs_ys):    
    for i in xs_ys:
        y, x = i
        chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\xs_vs_ys\{country}_{y}_vs_{x}.html'
        df = inputs_df[[x] + [y]].dropna(how='any')
        fig = go.Figure(data=[
             go.Scatter(x=df[x], y=df[y], mode='markers+text', text=df.index.month, textposition='top center')
         ])
        fig.write_html(chart_file, include_plotlyjs='cdn')  


def chart(country, title, df, latest_actual_date):
    chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\scaled_inputs_{country}_{title}.html'
    fig = go.Figure(data=[
             go.Scatter(name=x, x=df.index, y=df[x]) for x in df.columns             
         ])
    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(
        xaxis=dict(
                rangeselector=dict(
                        buttons=[
                                dict(count=7,
                                     label='1w',
                                     step='day',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='1m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=3,
                                     label='3m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=6,
                                     label='6m',
                                     step='month',
                                     stepmode='backward'),
                                dict(count=1,
                                     label='YTD',
                                     step='year',
                                     stepmode='todate'),
                                dict(count=1,
                                     label='1y',
                                     step='year',
                                     stepmode='backward'),
                                dict(step='all', label='All')
                            ]
                    ),
                rangeslider=dict(visible=True),
                type='date',
                dtick='M1'
            ),
        hovermode='x unified',
        title={
                'text' : title,
                'x' : 0.5,
                'yanchor' : 'top',
                'font_size' : 16
            })
    fig.write_html(chart_file, include_plotlyjs='cdn')   


def stacked_bar_chart(country, title, df, df2, latest_actual_date):
    chart_file = rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\scaled_inputs_stacked_bar_{country}_{title}.html'
    fig = go.Figure(data=[
             go.Bar(name=x, x=df.index, y=df[x], text=x) for x in df.columns             
         ])
    fig.add_scatter(name=df2.columns[0], x=df2.index, y=df2.iloc[:,0], line=dict(color='black', width=3))
    fig.add_scatter(name=df2.columns[1], x=df2.index, y=df2.iloc[:,1], line=dict(color='red', width=3))
    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(
        barmode='stack',
        hovermode='x unified',
        title={
                'text' : title,
                'x' : 0.5,
                'yanchor' : 'top',
                'font_size' : 16
            })
    fig.write_html(chart_file, include_plotlyjs='cdn')


def create_inputs_chart(country, forecast_df, x_parameters, y_parameter, latest_actual_date):
    standardized_inputs = pd.DataFrame(MinMaxScaler().fit_transform(forecast_df), columns=forecast_df.columns, index=forecast_df.index)
    chart(country, rf'standardized_inputs_{y_parameter}', standardized_inputs, latest_actual_date)    
    standardized_std_inputs = standardized_inputs[x_parameters].sum(axis=1)
    standardized_std_inputs = standardized_inputs[x_parameters].apply(lambda x: x / standardized_std_inputs) 
    for i in range(1, 13):
        i=1
        s = standardized_std_inputs[standardized_std_inputs.index.month == i]
        f = standardized_inputs[standardized_inputs.index.month == i][[y_parameter, y_parameter + '_predicted']]
        stacked_bar_chart(country, rf'standardized_inputs_{y_parameter}_{i}', s, f, latest_actual_date)
    

def bar_line_chart(country, df, df1, df2, title, primary_y_axis_title, latest_actual_date):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    data = list((df2).round(1))
    fig.add_bar(x=df2.index, y=df2, name=df2.name+ ' (left y axis)', text=list(data), secondary_y=False, textposition='inside', textfont=dict(color='black'))
    fig.add_trace(go.Scatter(x=df.index, y=df, name=df.name + ' (right y axis)', line=dict(color='black', width=3)), secondary_y=True)
    fig.add_trace(go.Scatter(x=df1.index, y=df1, name=df1.name + ' (right y axis)', line=dict(color='red', width=3)), secondary_y=True)
    fig.add_vline(x=latest_actual_date, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(title=title + ' last updated: ' + str(datetime.now()), xaxis_title='Date', yaxis_title=primary_y_axis_title, hovermode='x unified')
    fig.update_yaxes(title_text='mcm/d', secondary_y=True)
    fig.layout.yaxis.showgrid = False
    fig.write_html(rf'\\UK1-W-Z8-16\Users\swasti\Documents\Share\Japan\charts\{country}_{title}.html', include_plotlyjs='cdn')   
    