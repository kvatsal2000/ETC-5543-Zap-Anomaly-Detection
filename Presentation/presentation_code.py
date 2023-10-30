import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import date
import plotly_express as px
from sklearn.ensemble import IsolationForest
from datetime import date
import plotly_express as px
from sklearn.neighbors import LocalOutlierFactor


# Data Prepration
data = pd.read_csv('data.csv')
print(data)

DATE = []
for y, m in zip(data.reference_year, data.reference_month):
    DATE.append(date(y, m, 1))
data['DATE'] = DATE

data['DATE'] = pd.to_datetime(data['DATE'])

print(data.dtypes)


data_f = data[data['metric_name'] == "contracts_awarded_value" ]
print(data_f)

data_f1 = data[data['metric_name'] == "contracts_awarded_number" ]
print(data_f1)

data_after_2021 = data_f[data_f['DATE'] > '2021-08-01']

data_2021 = data_f[data_f['DATE'] <= '2021-08-31']


## full time series

sns.lineplot(data=data_f.groupby(['DATE', 'DTP_group'])['metric_value'].sum().reset_index(), x='DATE', y='metric_value', hue='DTP_group')
plt.title("Full Time Series")
plt.show()


## time series after Aug 2021


sns.lineplot(data=data_after_2021.groupby(['DATE', 'DTP_group'])['metric_value'].sum().reset_index(), x='DATE', y='metric_value', hue='DTP_group')
plt.title("Full Time Series After Aug 2021")
plt.show()


# using facets for group

new_group = data_after_2021.groupby(['DATE', 'DTP_group'])['metric_value'].sum().reset_index()
fig_g = px.line(new_group, facet_col='DTP_group', facet_col_wrap=5, x="DATE", y='metric_value')
fig_g = fig_g.update_yaxes(matches=None, showticklabels=True).update_xaxes(matches=None, showticklabels=True)
fig_g.show()


# IF
value_group = data_after_2021[['metric_value', 'DTP_group', 'date']]
dtp_group2 = np.unique(value_group['DTP_group'])
df = pd.DataFrame()
for group in dtp_group2:
    temp = None
    temp = value_group[value_group['DTP_group'] == group]
    #print(temp)
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.025),max_features=1.0)
    #print(model)
    model.fit(temp[['metric_value']])
    temp['scores_if']=model.decision_function(temp[['metric_value']])
    temp['anomaly_if']=model.predict(temp[['metric_value']])
    df = pd.concat([df, temp])

print(df)


fig1 = px.scatter(df,facet_col='DTP_group', facet_col_wrap=2, y='metric_value', x ='date', color='anomaly_if', title="Isolation Forest for Group Aggregation")
fig1.update_yaxes(matches=None, showticklabels = True).update_xaxes(matches=None, showticklabels = True)
fig1.show()


# LOF

value_group = data_after_2021[['metric_value', 'DTP_group', 'date']]
dtp_group2 = np.unique(value_group['DTP_group'])
df1 = pd.DataFrame()
for group in dtp_group2:
    temp = None
    temp = value_group[value_group['DTP_group'] == group]
    #print(temp)
    model = LocalOutlierFactor( n_neighbors=30, contamination=float(0.025))
    tempy = np.array(temp['metric_value'])
    tempy = tempy.reshape(-1,1)
    temp['anomaly_lof']=model.fit_predict(tempy)
    temp['scores_lof']=model.negative_outlier_factor_
    df1 = pd.concat([df1, temp])
print(df1)

fig2 = px.scatter(df1,facet_col='DTP_group', facet_col_wrap=2, y='metric_value', x ='date', color='anomaly_lof', title="Local Outlier Factor for Group Aggregation")
fig2.update_yaxes(matches=None, showticklabels = True).update_xaxes(matches=None, showticklabels = True)
fig2.show()



# combining

final_df = df.merge(df1, on=['metric_value','DTP_group','date'],how='left')
#print(final_df)


for index, row in final_df.iterrows():
        if row['anomaly_lof'] == -1 and row['anomaly_if'] == -1:
            final_df.at[index,'anomaly_final'] = -1
        #elif row['anomaly_LOF'] == -1 and row['anomaly'] == -1:
        else:
            final_df.at[index,'anomaly_final'] = 1
print(final_df)

fig3 = px.scatter(final_df,facet_col='DTP_group', facet_col_wrap=2, y='metric_value', x ='date', color='anomaly_final',title="Local Outlier Factor and Isolation forest combined")
fig3.update_yaxes(matches=None, showticklabels = True).update_xaxes(matches=None, showticklabels = True)
fig3.show()


## sestivity of contamination factor

value_group = data_after_2021[['metric_value', 'DTP_group', 'date']]
dtp_group2 = np.unique(value_group['DTP_group'])
df3 = pd.DataFrame()
for group in dtp_group2:
    temp = None
    temp = value_group[value_group['DTP_group'] == group]
    #print(temp)
    model = LocalOutlierFactor( n_neighbors=30, contamination=float(0.25))
    tempy = np.array(temp['metric_value'])
    tempy = tempy.reshape(-1,1)
    temp['anomaly_lof']=model.fit_predict(tempy)
    temp['scores_lof']=model.negative_outlier_factor_
    df3 = pd.concat([df3, temp])
print(df3)

fig2 = px.scatter(df3,facet_col='DTP_group', facet_col_wrap=2, y='metric_value', x ='date', color='anomaly_lof', title="Local Outlier Factor for Group Aggregation")
fig2.update_yaxes(matches=None, showticklabels = True).update_xaxes(matches=None, showticklabels = True)
fig2.show()
