from IF_function import metrics
from LOF1 import group_agg
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import date
import math


agg = ['group', 'group_division', 'group_region', 'contract_category'] #naming all aggregation levels
data = pd.read_csv('contract_awarded.csv') # loading data
data1 = pd.read_csv('contractors.csv') # loading data
data2 = pd.read_csv('splitting.csv') # loading data
data3 = pd.read_csv('metric_publication.csv') # loading data
data4 = pd.read_csv('po_split.csv') # loading data
data5 = pd.read_csv('process_integrity.csv') # loading data
data6 = pd.read_csv('ltd_comp.csv') # loading data
dataa = [data,data1,data2,data3,data4,data5,data6]


def anomaly_detection(metric,agg,contamination):
    
    col_names = metric.columns.tolist()
    forest = metrics(data=metric,aggregation=agg,cont_factor=contamination)
    outlier = group_agg(dataset=metric,aggregation=agg,cont_factor=contamination).rename(columns= {'anomaly':'anomaly_LOF','scores':'scores_LOF', 'percent': 'percent_LOF'})

    
        

    final_df = forest.merge(outlier, on = col_names ,how='left')
    final_df['anomaly_final'] = None
    for index, row in final_df.iterrows():

        if row['anomaly_LOF'] == 1 and row['anomaly'] == 1:

            final_df.at[index,'anomaly_final'] = 1

        else:

            final_df.at[index,'anomaly_final'] = -1
        
    return(final_df[final_df['anomaly_final'] == -1])

#func test
#anomaly_detection(metric=data5,agg='group_division',contamination=float(0.025)) 
#forest = metrics(data=data6,aggregation='group_division',cont_factor=float(0.025))
#outlier = group_agg(dataset=data6,aggregation='group_division',cont_factor=float(0.025))

qw2 = pd.DataFrame()

for q in agg:

    qw = pd.DataFrame()

    try:
        qw = anomaly_detection(metric=data,agg= q , contamination= float(0.025))
        print(qw)
    except AttributeError:
        continue
    

print(qw2)