

# import imaplib
# import email
# from email.header import decode_header
import os
from datetime import datetime, timedelta
import pandas as pd
# from zipfile import ZipFile
import pyodbc

#import subprocess
import numpy as np
import logging #version 0.5.1.2
from logging.handlers import RotatingFileHandler
# import psycopg2
from sqlalchemy import create_engine
#from pathlib import Path
# from Connections_database import connections 
import time
import sys
import gc
from sklearn.preprocessing import StandardScaler
#################################################################




folderName = os.path.dirname(__file__)
#folderName=os.path.dirname(folderName)
            
class furnace_deviation_tracker():
    
    def __init__(self):
        try:
            self.startTime = datetime.now()        
            self.logFile = os.path.join(folderName, "runLog_furnace_deviation.log")
            logging.basicConfig(filename=self.logFile, filemode='a', level=logging.INFO ,format='%(asctime)s: %(message)s', datefmt='%d %b %y %H:%M:%S')
            self.logData("## Starting Script ## ")

            self.config_df=pd.read_csv(os.path.join(folderName,'config','config.csv'))
            
            self.last_pivot_df=pd.DataFrame()
            self.lbt_df=pd.read_csv(os.path.join(folderName,'config','lbt_parameters.csv'))
            
            self.save_after_iteration_number=self.config_df['save_after_iteration_number'].values[0]
            
            
            self.file_to_run=self.config_df['file_to_run'].values[0]
            self.input_df=pd.read_csv(os.path.join(folderName,'input',str(self.file_to_run)+'.csv'))
            
            self.clean_file=self.config_df['clean_file'].values[0]
            self.clean_df=pd.read_csv(os.path.join(folderName,'input',str(self.clean_file)+'.csv'))
            
            
            self.tags_compare_pivot=self.lbt_df['compare_pivot'].dropna().tolist()
            #self.input_df=pd.read_csv(os.path.join(folderName,'input','input.csv'))
                        
            self.input_df['time']=pd.to_datetime(self.input_df['time'],format='%Y-%m-%d %H:%M:%S')


        except Exception as e:
            self.logData('problem in initilization'+str(e))
            

        
        try:
            self.input_df['time']=pd.to_datetime(self.input_df['time'],format='%Y-%m-%d %H:%M:%S')
            self.input_df=self.input_df.sort_values(by=['time'],ascending=False)
            self.input_df=self.input_df.reset_index(drop=True)
        except:
            self.logData('problem in input sheet date time format')

        try:
            self.clean_df['time']=pd.to_datetime(self.clean_df['time'],format='%Y-%m-%d %H:%M:%S')
            self.clean_df=self.clean_df.sort_values(by=['time'],ascending=False)
            self.clean_df=self.clean_df.reset_index(drop=True)
        except:
            self.logData('problem in clean_df sheet date time format')


    def logData(self, statusText):
        currTime = (datetime.now())
        #currTime = (datetime.now() - self.startTime)
        print ("{} : {}".format(str(currTime)[:-7], statusText))
        logging.info(statusText)
        
    
    def generate_lbt(self,input_df_current_time):
        #print('generating lbt')
        lbt_df=self.lbt_df[['tag','min','max']].dropna()
        lbt_df['min_value']=np.nan
        lbt_df['max_value']=np.nan
        clean_df=self.clean_df
        filtered_df=pd.DataFrame(columns=['time'])
        
        for i,row in lbt_df.iterrows():
            curr_formulae_min=str(row['min'])
            curr_formulae_max=str(row['max'])
            temp_formulae_min=curr_formulae_min
            temp_formulae_max=curr_formulae_max

            tags_in_formulae_min=[] 
            tags_in_formulae_max=[] 

            while temp_formulae_min.find("{") != -1:
                tags_in_formulae_min.append(temp_formulae_min[temp_formulae_min.find("{")+len("{"):temp_formulae_min.find("}")])
                temp_formulae_min = temp_formulae_min[temp_formulae_min.find("}")+1:] 
                
            while temp_formulae_max.find("{") != -1:
                tags_in_formulae_max.append(temp_formulae_max[temp_formulae_max.find("{")+len("{"):temp_formulae_max.find("}")])
                temp_formulae_max = temp_formulae_max[temp_formulae_max.find("}")+1:] 

            
            for tags in tags_in_formulae_min:
                curr_formulae_min = curr_formulae_min.replace(str("{"+tags+"}"),str("input_df_current_time['"+str(tags)+"'].values[0]"))
                try:
                    a=eval(str("input_df_current_time['"+str(tags)+"'].values[0]"))
                except:
                    self.logData('problem in tag:- '+str(tags))  
                    
            for tags in tags_in_formulae_max:
                curr_formulae_max = curr_formulae_max.replace(str("{"+tags+"}"),str("input_df_current_time['"+str(tags)+"'].values[0]"))
                try:
                    a=eval(str("input_df_current_time['"+str(tags)+"'].values[0]"))
                except:
                    self.logData('problem in tag:- '+str(tags))  
                    

            try:
                a=eval(curr_formulae_min)
                lbt_df.loc[i,'min_value']=eval(curr_formulae_min)
                lbt_df.loc[i,'max_value']=eval(curr_formulae_max)
            except Exception as e:
                self.logData('problem in evaluating formulae of'+str(curr_formulae)+'     ' +str(e))
                

        filtered_df['time']=input_df_current_time['time']
        for i,row in lbt_df.iterrows():
            clean_df=clean_df[clean_df[str(row['tag'])]<row['max_value']]
            filtered_df.loc[0,str(row['tag']+'_max')]=clean_df.shape[0]
            
            clean_df=clean_df[clean_df[str(row['tag'])]>row['min_value']]
            filtered_df.loc[0,str(row['tag']+'_min')]=clean_df.shape[0]
            
        
        
        obj_function_df=self.lbt_df[['tag_performance','value']].dropna()
        
        # closest to pivot
        if clean_df.shape[0]>0:
            current_pivot_df=clean_df[self.tags_compare_pivot]
            current_pivot_df_values=current_pivot_df.values
            scaled_current_pivot_df_values=self.scaler_test.transform(current_pivot_df_values)
        
        
        if self.last_pivot_df.shape[0]==0 and clean_df.shape[0]>0:
            if obj_function_df['value'].values[0]=='min':
                clean_df=clean_df.sort_values(by=str(obj_function_df['tag_performance'].values[0]),ascending=True)
                clean_df=clean_df.reset_index(drop=True)
            elif obj_function_df['value'].values[0]=='max':
                clean_df=clean_df.sort_values(by=str(obj_function_df['tag_performance'].values[0]),ascending=False)
                clean_df=clean_df.reset_index(drop=True)
        
        elif self.last_pivot_df.shape[0]>0 and clean_df.shape[0]>0  :
            #calculate distance
            a=scaled_current_pivot_df_values-self.scaled_last_pivot_df_values
            a_sum=a.sum(axis=1)
            df=pd.DataFrame(a)
            df=abs(df)
            df['sum']=a.sum(axis=1)
            df['sum']=df['sum'].abs()
            df=df.sort_values(by='sum',ascending=True)
            clean_df=clean_df.reset_index(drop=True)
            clean_df=clean_df.iloc[df.head(1).index.values[0]:df.head(1).index.values[0]+1,:]
            
            
            
            
            
        pivot_df=clean_df.head(1)
        
        
        
        if pivot_df.shape[0]>0:
            pivot_df=pivot_df.rename(columns={'time':'historical_time'})
            pivot_df.insert(0,'time',input_df_current_time['time'].values[0])
            pivot_df=pivot_df.reset_index(drop=True)
            self.last_pivot_df=pivot_df[self.tags_compare_pivot]
            self.last_pivot_df_values=pivot_df[self.tags_compare_pivot].values
            self.scaled_last_pivot_df_values=self.scaler_test.transform(self.last_pivot_df_values)
            
            
        if pivot_df.shape[0]==0:
            self.last_pivot_df=pivot_df[self.tags_compare_pivot]
            pivot_df=pivot_df.rename(columns={'time':'historical_time'})
            pivot_df.insert(0,'time',input_df_current_time['time'].values[0]) 
            pivot_df['time']=input_df_current_time['time']
            pivot_df=pivot_df.reset_index(drop=True)
            
            
        z_score_raw_df,z_score_weighted_df=self.generate_z_scores(input_df_current_time,pivot_df)
        return pivot_df,filtered_df,z_score_raw_df,z_score_weighted_df
    
    
    
    def generate_z_scores(self,input_df_current_time,pivot_df):
        clean_df=self.clean_df
        z_score_tags=self.lbt_df['calculate_z_score'].dropna().to_list()
        
        mean_clean_df=clean_df[z_score_tags].mean()
        std_dev_clean_df=clean_df[z_score_tags].std()
        

        z_score_df=pd.DataFrame()
        
        for tag in z_score_tags:
            z_score_df.loc[0,tag]=(input_df_current_time[tag].values[0]-pivot_df[tag].values[0])/std_dev_clean_df[tag]
            
        # for tag in z_score_tags:
        #     z_score_df.loc[0,tag]=(input_df_current_time[tag].values[0]-mean_clean_df[tag])/std_dev_clean_df[tag]    
            
        # for tag in z_score_tags:
        #     z_score_df.loc[0,tag]=(input_df_current_time[tag].values[0]-pivot_df[tag].values[0])/std_dev_clean_df[tag]
            
            
        # z_score_df_abs=z_score_df.abs()
        # sum_z_score=z_score_df_abs.sum(axis=1)
        
        
        
        df_=self.lbt_df[['calculate_z_score','good','weights']].dropna()
        z_score_df=z_score_df.melt()
        z_score_df['value_good']=z_score_df['value']*df_['good']
        z_score_df['value_good_weighted']=z_score_df['value_good']*df_['weights']
        z_score_df['value_good_weighted_abs']=z_score_df['value_good_weighted'].abs()
        
        sum_weighted_z_score=z_score_df['value_good_weighted_abs'].sum()     
        
        z_score_df['value_good_weighted_normalized']=z_score_df['value_good_weighted']/sum_weighted_z_score
        z_score_df['value_good_weighted_normalized_percentage']=z_score_df['value_good_weighted_normalized']*100
        
        
        z_score_df['a']=0 
        
        z_score_weighted_df=z_score_df.pivot(index='a',columns='variable',values='value_good_weighted_normalized_percentage')
        z_score_raw_df=z_score_df.pivot(index='a',columns='variable',values='value') 
        
        
        z_score_weighted_df.insert(0,'time',input_df_current_time['time'].values[0])
        z_score_raw_df.insert(0,'time',input_df_current_time['time'].values[0])
        
        
        return z_score_raw_df,z_score_weighted_df
    
    def calculate_rolling_average_of_test_data(self,input_df):
        rol_avg_tags_df=self.lbt_df[['calculate rolling average']].dropna()
        rol_avg_days=self.lbt_df['rolling_average_days'].dropna().values[0].astype(int)
        rol_avg_days=str(rol_avg_days)+'D'
        input_df=input_df.sort_values(by='time',ascending=True)
        input_df = input_df.set_index('time')
        input_df=input_df.apply(pd.to_numeric,errors='coerce')
        for tag in rol_avg_tags_df['calculate rolling average']:            
            input_df[tag+'_rol_avg']=input_df[tag].rolling(rol_avg_days).mean() 
        
        input_df.insert(0,'time',input_df.index)
        input_df=input_df.reset_index(drop=True)
        input_df.to_csv(os.path.join(folderName,'output','rolling_avg_of_test_data_'+str(self.file_to_run)+'.csv'),index=False) 
        

        
        return input_df
    
    
    def calculate_rolling_average_of_clean_data(self,clean_df):
        
        rol_avg_tags_df=self.lbt_df[['calculate rolling average']].dropna()
        rol_avg_days=self.lbt_df['rolling_average_days'].dropna().values[0].astype(int)
        rol_avg_days=str(rol_avg_days)+'D'
        
        
        clean_df=clean_df.sort_values(by='time',ascending=True)
        clean_df = clean_df.set_index('time')
        clean_df=clean_df.apply(pd.to_numeric,errors='coerce')
        for tag in rol_avg_tags_df['calculate rolling average']:            
            clean_df[tag+'_rol_avg']=clean_df[tag].rolling(rol_avg_days).mean()     
        clean_df.insert(0,'time',clean_df.index)
        clean_df=clean_df.reset_index(drop=True)
        
        clean_df.to_csv(os.path.join(folderName,'output','rolling_avg_of_clean_data_'+str(self.file_to_run)+'.csv'),index=False) 
        
        input_X=clean_df[self.tags_compare_pivot]

        input_X=input_X.values
        self.scaler_test = StandardScaler()
        self.scaler_test.fit(input_X)
        
        return clean_df      

        

    
    def main_function(self):
        input_df=self.input_df
        input_df=self.calculate_rolling_average_of_test_data(input_df)
        self.clean_df=self.calculate_rolling_average_of_clean_data(self.clean_df)
 
        
        points_to_run=int(self.config_df['pointsToRun'].values[0])
        if points_to_run>(input_df.shape[0]):
            points_to_run=(input_df.shape[0])        
            if points_to_run<1:
                points_to_run=int(self.config_df['pointsToRun'].values[0])

        file_to_run=self.config_df['file_to_run'].values[0]
        output_final=pd.DataFrame()
        final_pivot_df=pd.DataFrame()
        final_filtered_df=pd.DataFrame()

        final_z_score_raw_df=pd.DataFrame()
        final_z_score_weighted_df=pd.DataFrame()
        

        
        for i in range(0,points_to_run):
            try:  
                print(i)
                if i==13:
                    print('stop')
                input_df_current_time=input_df.iloc[[i]].copy(deep=True)
                input_df_current_time=input_df_current_time.reset_index(drop=True)
                
                pivot_df,filtered_df,z_score_raw_df,z_score_weighted_df=self.generate_lbt(input_df_current_time.head(1))
                
                try:
                    final_pivot_df=final_pivot_df.append(pivot_df, ignore_index=True)
                    final_filtered_df=final_filtered_df.append(filtered_df, ignore_index=True)
                    final_z_score_raw_df=final_z_score_raw_df.append(z_score_raw_df, ignore_index=True)
                    final_z_score_weighted_df=final_z_score_weighted_df.append(z_score_weighted_df, ignore_index=True)
                except Exception as e:
                    self.logData('problem in appending pivot, or final_filtered   '+str(e))
            except Exception as e:
                self.logData('problem in main function   '+str(e))
            
            
            final_pivot_df['time']=pd.to_datetime(final_pivot_df['time'],format='%Y-%m-%d %H:%M:%S')
            final_filtered_df['time']=pd.to_datetime(final_filtered_df['time'],format='%Y-%m-%d %H:%M:%S')
            
            
            final_pivot_df=final_pivot_df.sort_values(by=['time'],ascending=True)
            final_filtered_df=final_filtered_df.sort_values(by=['time'],ascending=True)
            
            
            
            
            
            if i % self.save_after_iteration_number== 0 and i>0:
                final_pivot_df.to_csv(os.path.join(folderName,'output','pivot_data_'+str(self.file_to_run)+'.csv'),index=False)
                final_filtered_df.to_csv(os.path.join(folderName,'output','filtering_rows_'+str(self.file_to_run)+'.csv'),index=False)
                final_z_score_raw_df.to_csv(os.path.join(folderName,'output','z_scores_raw_'+str(self.file_to_run)+'.csv'),index=False)
                final_z_score_weighted_df.to_csv(os.path.join(folderName,'output','z_scores_weighted_'+str(self.file_to_run)+'.csv'),index=False)
        final_pivot_df.to_csv(os.path.join(folderName,'output','pivot_data_'+str(self.file_to_run)+'.csv'),index=False)
        final_filtered_df.to_csv(os.path.join(folderName,'output','filtering_rows_'+str(self.file_to_run)+'.csv'),index=False)
        final_z_score_raw_df.to_csv(os.path.join(folderName,'output','z_scores_raw_'+str(self.file_to_run)+'.csv'),index=False)
        final_z_score_weighted_df.to_csv(os.path.join(folderName,'output','z_scores_weighted_'+str(self.file_to_run)+'.csv'),index=False) 

            
        # output_final=output_final.sort_values(by=['time'],ascending=True)
        
        
        
        
        # output_final.to_csv(os.path.join(folderName,'output','output_algo_'+str(self.file_to_run)+'.csv'),index=False)
        
            
            



furnace_deviation_tracker_obj=furnace_deviation_tracker()
furnace_deviation_tracker_obj.main_function()



