import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import plotly.express as px
from datetime import date
import math
import re
import warnings




data = pd.read_csv('contract_awarded.csv') # loading data
#cont = ''.join([col for col in data if col.startswith('contract')]) #different datasets have different col names for contract_category

agg = ['group', 'group_division', 'group_region', 'contract_category'] #naming all aggregation levels

warnings.filterwarnings('ignore')
def metrics(data, aggregation, cont_factor): # dataset, aggregation and contamination factor parameters
    DATE = []
    for y, m in zip(data.reference_year, data.reference_month): #reference_month, year e.t.c are all strings, so converting them to date type
        if math.isnan(y) or math.isnan(m):
            DATE.append(None)
            continue
        else:
            y = int(y)
            m = int(m)
            DATE.append(date(y, m, 1))

    data['date'] = DATE
    data['date'] = pd.to_datetime(data['date'])
    data = data[data['date'] > "2021-08-01"]  
    cols = set(list(data.columns)) #list of col names
    li = list(data['metric_name'].unique()) #list of different metric_names in a dataset
    group_region = set(['DTP_group', 'DTP_region']) #set created to check if region or division exists in the dataset
    group_division = set(['DTP_group', 'DTP_division'])
    cont = ''.join([col for col in data if col.startswith('contract')]) #different datasets have different col names for contract_category

    
    #group 


    if aggregation == "group": #if aggregation is group 
        df = data[0:0]
        
        group = list(data['DTP_group'].unique()) #get different groups in the dataset
        for x in li: #for each metric_name in a dataset
            

            data1 = data[data['metric_name'] == x] #filter data for that metric_name
            for y in group: #for each group in the filtered dataset
                temp = None
                temp = data1[data1['DTP_group'] == y] #filter data for group y
                
                if len(temp) == 0: #check if there are observations in the filtered dataset 
                    continue
                model = IsolationForest(n_estimators=50, max_samples='auto', contamination=cont_factor,max_features=1.0) #defining model
                if re.search('percentage', x) is not None or re.search('publication', x) is not None:  #check if the metric_name x has 'percentage' in it 
                    temp = temp.groupby('date').agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index() #group by date and calculate percentage by dividing sum(num) by sum(denominator) for group y
                    temp['percent']= (temp['metric_value_numerator']/temp['metric_value_denominator'])*100 #creating column to store percentage
                    model.fit(temp[['percent']]) #apply model to identify anomalies
                    temp['scores']=model.decision_function(temp[['percent']]) #storing scores and anomaly indicator in columns
                    temp['anomaly']=model.predict(temp[['percent']])
                    temp['DTP_group'] = y #retain name of group and metric_name so that we can identify an observation
                    temp['metric_name'] = x
                    
                    df = pd.concat([df, temp])
                    
                else: #if metric_name doesn't have 'percentage' in it
                    model.fit(temp[['metric_value']])
                    temp['scores']=model.decision_function(temp[['metric_value']])
                    temp['anomaly']=model.predict(temp[['metric_value']])
                    df = pd.concat([df, temp])
        return(df)

    #Contract_type/Contract_category
    

    elif aggregation == 'contract_category':
        
        df = data[0:0]
        df1 = pd.DataFrame()
        
        if cont != '': #check if contract_category exists
            contract = list(data[cont].unique()) #get different contract_category types
            
            for x in li: #for each metric_name in a dataset
            

                data1 = data[data['metric_name'] == x]
                for y in contract: #for each contract y in filtered data
                    temp = None
                    temp = data1[data1[cont] == y]
                    
                    model = IsolationForest(n_estimators=50, max_samples='auto', contamination=cont_factor,max_features=1.0)
                    if re.search('percentage', x) is not None or re.search('publication', x) is not None:  #check if the metric_name x has 'percentage' in it 
                        temp = temp.groupby('date').agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                        temp['percent']= (temp['metric_value_numerator']/temp['metric_value_denominator'])*100
                        model.fit(temp[['percent']])
                        temp['scores']=model.decision_function(temp[['percent']])
                        temp['anomaly']=model.predict(temp[['percent']])
                        temp[x] = y
                        temp['metric_name'] = x
                        df = pd.concat([df, temp])
                        
                    else: #if metric_name doesn't have 'percentage' in it
                        model.fit(temp[['metric_value']])
                        temp['scores']=model.decision_function(temp[['metric_value']])
                        temp['anomaly']=model.predict(temp[['metric_value']])
                        df = pd.concat([df, temp])
            return(df)
        else:
            return(None) #return none if contract_category column doesn't exist

    #group/division
    elif aggregation == 'group_division':
        df = data[0:0]
        df1 = pd.DataFrame()
        group = list(data['DTP_group'].unique()) #get unique groups 
        if group_division.issubset(cols): #to run the code for group/division aggregation we need to check if both columns exist
        
            division = list(data['DTP_division'].unique()) #get unique division names
            for x in li:
                data1 = data[data['metric_name'] == x] #filter data for metric_name x
                for y in group: #for group y in metric_name x
                    df = data[0:0]
                    data2 = data1[data1['DTP_group'] == y]
                    for z in division: #for division z in group y in metric_name x
                        temp = None
                        temp = data2[data2['DTP_division'] == z]
                        
                        if len(temp) == 0:#check if data set is empty or not
                            continue
                        
                        model = IsolationForest(n_estimators=50, max_samples='auto', contamination=cont_factor,max_features=1.0)
                        if re.search('percentage', x) is not None or re.search('publication', x) is not None:  #check if the metric_name x has 'percentage' in it 
                            temp = temp.groupby('date').agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                            temp['percent']= (temp['metric_value_numerator']/temp['metric_value_denominator'])*100
                            model.fit(temp[['percent']])
                            temp['scores']=model.decision_function(temp[['percent']])
                            temp['anomaly']=model.predict(temp[['percent']])
                            temp['DTP_group'] = y #retain names of group division and metric name for each observation after summarising results
                            temp['metric_name'] = x
                            temp['DTP_division'] = z
                            df = pd.concat([df, temp])
                        
                        else: #if metric_name doesn't have 'percentage' in it
                            model.fit(temp[['metric_value']])
                            temp['scores']=model.decision_function(temp[['metric_value']])
                            temp['anomaly']=model.predict(temp[['metric_value']])
                            df = pd.concat([df, temp])
                    df1 = pd.concat([df1,df])
            return(df1)  
        else:
            return(None) 
       
        

    #group/region
    elif aggregation == 'group_region':
    
        df = data[0:0]
        df1 = pd.DataFrame()
        group = list(data['DTP_group'].unique())
        if group_region.issubset(cols): #to run the code for group/region aggregation we need to check if both columns exist
            region = list(data['DTP_region'].unique()) #get unqiye region names
            for x in li: #for each metric_name in dataset
                data1 = data[data['metric_name'] == x] #filter data for metric_name x 
                for y in group: #for each group in filtered dataset
                    df = data[0:0]
                    data2 = data1[data1['DTP_group'] == y] #filter data for group y in metric_name x
                    for z in region:#for region z in group y in metric_name x
                        temp = None
                        temp = data2[data2['DTP_region'] == z]
                        if len(temp) == 0:
                            continue
                        
                        model = IsolationForest(n_estimators=50, max_samples='auto', contamination=cont_factor,max_features=1.0)
                        if re.search('percentage', x) is not None or re.search('publication', x) is not None:  #check if the metric_name x has 'percentage' in it 
                            temp = temp.groupby('date').agg({'metric_value_numerator': 'sum', 'metric_value_denominator': 'sum'}).reset_index()
                            temp['percent']= (temp['metric_value_numerator']/temp['metric_value_denominator'])*100
                            model.fit(temp[['percent']])
                            temp['scores']=model.decision_function(temp[['percent']])
                            temp['anomaly']=model.predict(temp[['percent']])
                            temp['DTP_group'] = y #retain important info
                            temp['metric_name'] = x
                            temp['DTP_region'] = z
                            
                            df = pd.concat([df, temp])
                        
                        else: #if metric_name doesn't have 'percentage' in it
                            model.fit(temp[['metric_value']])
                            temp['scores']=model.decision_function(temp[['metric_value']])
                            temp['anomaly']=model.predict(temp[['metric_value']])
                            df = pd.concat([df, temp])
                    df1 = pd.concat([df1,df])
            return(df1) 
        else:
            return(None) 
    
    