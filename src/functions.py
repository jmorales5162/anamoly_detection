import Config as cfg
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fig_prod_month():
    """
    Esto muestra la produccion mensual de energia
    """
    weekly_production = cfg.dt_p1_gen.resample('W-Mon', on='DATE_TIME').sum(numeric_only=True)
      
    fig = px.bar(weekly_production, x=weekly_production.index, y="DAILY_YIELD", 
                title='Weekly Production', 
                labels={'DAILY_YIELD': 'Total Production (kW)', 'index': 'Date'},
                color="DAILY_YIELD",  
                color_continuous_scale="Viridis",  
                hover_name=weekly_production.index,  
                hover_data={"DAILY_YIELD": True} 
                )

    fig.update_layout(
        plot_bgcolor="white", 
        title_font=dict(size=24, family="Arial", color="black"),  
        xaxis_title_font=dict(size=18, family="Arial", color="grey"),  
        yaxis_title_font=dict(size=18, family="Arial", color="grey"),  
        xaxis_tickfont=dict(size=12, family="Arial", color="grey"),  
        yaxis_tickfont=dict(size=12, family="Arial", color="grey"),  
        bargap=0.1  
    )

    fig.show()

def prod_week():
    cfg.dt_p1_sens['DAY_OF_WEEK'] = cfg.dt_p1_sens['DATE_TIME'].dt.dayofweek + 1
    cfg.dt_p1_sens['HOUR'] = cfg.dt_p1_sens['DATE_TIME'].dt.hour

    pivot_temp = cfg.dt_p1_sens.groupby(['HOUR', 'DAY_OF_WEEK'])['AMBIENT_TEMPERATURE'].mean().unstack()

    plt.figure(figsize=(20, 8))
    sns.heatmap(pivot_temp, cmap="YlGnBu", annot=True, fmt=".1f", cbar_kws={'label': 'Ambient Temperature (°C)'})
    plt.title('Hourly Average Ambient Temperature by Day of Week')
    plt.xlabel('Day of Week (1=Monday, 7=Sunday)')
    plt.ylabel('Hour of Day')
    plt.show()

def matrix_corr():
    cfg.dt_p1_sens.set_index('DATE_TIME', inplace=True)
    cfg.dt_p1_gen.set_index('DATE_TIME', inplace=True)

    merged_data = cfg.dt_p1_sens[['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']].merge(
        cfg.dt_p1_gen[['DC_POWER', 'AC_POWER', 'DAILY_YIELD']], 
        left_index=True, 
        right_index=True,
        how='inner'
    )

    correlation_matrix_english = merged_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix_english, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Variables')
    plt.show()

