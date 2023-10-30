import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import date
from sklearn.neighbors import LocalOutlierFactor
import math
import re
import warnings



aggregation = ['group','group_division','group_region','contract_category'] # list of different types of aggregation available in the tables

warnings.filterwarnings('ignore') # to ignore the warnings

def group_agg(dataset,aggregation, cont_factor):      # defining function with 3 parameters i.e. dataset , aggregation and contamination factor

    DATE = []
    dataset['reference_year'] = pd.to_numeric(dataset['reference_year'])  
    dataset['reference_month'] = pd.to_numeric(dataset['reference_month'])

    DATE = list()
    for y, m in zip(dataset.reference_year, dataset.reference_month):
        if math.isnan(y) or math.isnan(m):
            DATE.append(None)
            continue
        else:
            y = int(y)
            m = int(m)
            DATE.append(date(y, m, 1))    # converting reference month and year to datetype

    dataset['date'] = DATE       
    dataset['date']=pd.to_datetime(dataset['date'])
    dataset = dataset[dataset['date'] > "2021-08-01"]  

    col = set(list(dataset.columns))       # set of columns in the dataset 
    re_list = set(['DTP_group','DTP_region'])  # set with DTP_group, DTP_region
    di_list = set(['DTP_group','DTP_division'])  # set with DTP_group, DTP_division



    # group
    if aggregation == 'group':   # for aggregation level : group
            datag = dataset
            group = np.unique(datag['DTP_group']) # list of groups in the dataset
            met = np.unique(datag['metric_name']) # list of metric name in the dataset
            df_finala = dataset[0:0]
            gpd = None                             # creating empty datasets.
            for z in met:                          # looping over every metric name 
                data1 = None
                data1 = datag[datag['metric_name'] == z]
                for x in group:                     # looping over every individual group
                    tempn = None
                    tempn = data1[data1['DTP_group'] == x]
                    if len(tempn) <= 1:              # checking if the the nuumber of observations in the group are atleast 1. 
                        continue                      #if its less than 1, the inner loop terminates 
                    lof = LocalOutlierFactor(n_neighbors=1, contamination= cont_factor)  # fitting the LOF Model
                    if re.search('percentage', z) is not None or re.search('publication', z) is not None:   # checking if metric name is having percentage then adding the denominator and numerator.

                        # calculating anomaly in percentage by month.
                        # grouping by 'metric_name','DTP_group','date' to retain the names and values of the column making it identifiable.
                        gpd = tempn.groupby(['metric_name','DTP_group','date']).agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                        gpd['percent'] = (gpd['metric_value_numerator']/gpd['metric_value_denominator'])*100
                        if len(gpd) <= 1:
                            continue
                        tempy = np.array(gpd['percent'])
                        tempy = tempy.reshape(-1,1)       # converting the value in 2-D array as its the input requirement for lof.fit.predict.
                        gpd['anomaly'] = lof.fit_predict(tempy)  # predicting the anomaly 
                        gpd['scores'] = lof.negative_outlier_factor_ # calculating the scores
                        df_finala = pd.concat([df_finala,gpd])
                    else:
                        # if the metric name does not have percentage.
                        tempy = np.array(tempn['metric_value'])
                        tempy = tempy.reshape(-1,1)
                        tempn['anomaly'] = lof.fit_predict(tempy)
                        tempn['scores'] = lof.negative_outlier_factor_
                        df_finala = pd.concat([df_finala,tempn])
            return df_finala         
            #print(df_finala.head())


    # group /division
    elif aggregation == 'group_division':  # aggeration level: group_division
        if di_list.issubset(col):         # checking if the datset has division or not. 
            datag = dataset
            dtp_group = np.unique([datag['DTP_group']])
            matrices = list(np.unique([datag['metric_name']]))
            division = list(datag['DTP_division'].unique())    # list of divisions in the dataset
            df = pd.DataFrame()
            df1 = pd.DataFrame()

            df2 = dataset[0:0]
            gpd = None
            for y in matrices:                          # looping over every metric name 
                df = datag[datag['metric_name'] == y]
                for x in dtp_group:                     # looping over every individual group
                    data2 = df[df['DTP_group'] == x] 
                    df2 = dataset[0:0]
                    for z in division:                  # looping over every individual division 
                        tempn = None
                        tempn = data2[data2['DTP_division'] == z]
                        if len(tempn) <= 1:
                            continue
                        lof = LocalOutlierFactor(n_neighbors=1, contamination= cont_factor)
                        if re.search('percentage', y) is not None or re.search('publication', y) is not None:
                            gpd = tempn.groupby(['metric_name','DTP_group','date','DTP_division']).agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                            gpd['percent'] = (gpd['metric_value_numerator']/gpd['metric_value_denominator'])*100
                            if len(gpd) <= 1:
                                continue
                            tempy = np.array(gpd['percent'])
                            tempy = tempy.reshape(-1,1)
                            gpd['anomaly'] = lof.fit_predict(tempy)
                            gpd['scores'] = lof.negative_outlier_factor_
                            df2 = pd.concat([df2,gpd])
                        else:
                            tempy = np.array(tempn['metric_value'])
                            tempy = tempy.reshape(-1,1)
                            tempn['anomaly'] = lof.fit_predict(tempy)
                            tempn['scores'] = lof.negative_outlier_factor_
                            df2 = pd.concat([df2,tempn])   
                    df1 = pd.concat([df1,df2])
            return df1
            #print(df1.head(10)) 
        else:
            return None


    # group/region

    elif aggregation == 'group_region':  # aggregation level : group_region
        if re_list.issubset(col):         # checking if the datset has region or not.
            datag = dataset
            dtp_group = np.unique([datag['DTP_group']])
            matrices = list(np.unique([datag['metric_name']]))
            region = list(datag['DTP_region'].unique())
            final2 = pd.DataFrame()
            df_r = pd.DataFrame()
            df2 = dataset[0:0]
            gpd = None
            for y in matrices:                       # looping over every metric name 
                data1 = datag[datag['metric_name'] == y]
                for x in dtp_group:                   # looping over every individual group
                    data2 = data1[data1['DTP_group'] == x] 
                    df2 = dataset[0:0]
                    for z in region:                   # looping over every individual division 
                        tempn = None
                        tempn = data2[data2['DTP_region'] == z]
                        if len(tempn) <= 1:
                            continue
                        lof = LocalOutlierFactor(n_neighbors=1, contamination= cont_factor)
                        if re.search('percentage', y) is not None or re.search('publication', y) is not None:
                            gpd = tempn.groupby(['metric_name','DTP_group','date','DTP_region']).agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                            gpd['percent'] = (gpd['metric_value_numerator']/gpd['metric_value_denominator'])*100
                            if len(gpd) <= 1:
                                continue
                            tempy = np.array(gpd['percent'])
                            tempy = tempy.reshape(-1,1)
                            gpd['anomaly'] = lof.fit_predict(tempy)
                            gpd['scores'] = lof.negative_outlier_factor_
                            df2 = pd.concat([df2,gpd])
                        else:
                            tempy = np.array(tempn['metric_value'])
                            tempy = tempy.reshape(-1,1)
                            tempn['anomaly'] = lof.fit_predict(tempy)
                            tempn['scores'] = lof.negative_outlier_factor_
                            df2 = pd.concat([df2,tempn])
                    df_r = pd.concat([df_r,df2])
            return df_r
            #print(df_r.head(10))   
        else:
            return None

    #contract_type
    elif aggregation == 'contract_category':
        df = dataset[0:0]
        df1 = pd.DataFrame()
        gpd = None
        cont = ''.join([col for col in dataset if col.startswith('contract')]) # different datasets heve different names for contract catrgory or type 
        if cont != '':                                      # checking if cont existis i.e. if contract category exists in the dataset.
            contract = list(dataset[cont].unique())         # list of contract category.
            li = list(dataset['metric_name'].unique())
            for x in li:
                data1 = dataset[dataset['metric_name'] == x]
                for y in contract:
                    temp = None
                    tempn = data1[data1[cont] == y]
                    lof = LocalOutlierFactor(n_neighbors=1, contamination= cont_factor)
                    if re.search('percentage', x) is not None or re.search('publication', x) is not None: 
                        gpd = tempn.groupby(['date','metric_name']).agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                        gpd['percent'] = (gpd['metric_value_numerator']/gpd['metric_value_denominator'])*100
                        if len(gpd) <= 1:
                                continue
                        tempy = np.array(gpd['percent'])
                        tempy = tempy.reshape(-1,1)
                        gpd['anomaly'] = lof.fit_predict(tempy)
                        gpd['scores'] = lof.negative_outlier_factor_
                        gpd['Contract_type'] = y
                        df = pd.concat([df, gpd])
                    else:
                        tempy = np.array(tempn['metric_value'])
                        tempy = tempy.reshape(-1,1)
                        tempn['anomaly'] = lof.fit_predict(tempy)
                        tempn['scores'] = lof.negative_outlier_factor_
                        df = pd.concat([df, tempn])
            return df
            #print(df.head(10))
        else:
            return None




#group_agg(data_ltd,'group_region')




#for xv in aggregation:

#    if group_agg(dataset=data_ltd, aggregation=xv) is None:

#        print('none' + " " + xv)

#        continue

#    else:

#        group_agg(dataset=data_ltd, aggregation=xv).to_csv('tryout' + '_' + xv + '.csv')

