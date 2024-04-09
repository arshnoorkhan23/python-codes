
import imaplib
import email
from email.header import decode_header
import os
from datetime import datetime, timedelta
import pandas as pd
from zipfile import ZipFile
import pyodbc
from fast_to_sql import fast_to_sql as fts
import subprocess
import numpy as np
import logging #version 0.5.1.2
from logging.handlers import RotatingFileHandler
import psycopg2
from sqlalchemy import create_engine
from pathlib import Path
from service_raw_data import SMWinservice


folderName = os.path.dirname(__file__)
logging.basicConfig(filename=os.path.join(folderName,'log_file_for_1_min_aggregator.log'), filemode='a', level=logging.INFO ,format='%(asctime)s: %(message)s', datefmt='%d %b %y %H:%M:%S')
def logData(statusText):
    currTime = (datetime.now())
    print ("{} : {}".format(str(currTime)[:-7], statusText))
    logging.info(statusText)

# tf='2020-12-02 08:30:00'
# tf=datetime.strptime(tf,'%Y-%m-%d %H:%M:%S')
iteration=0

def connecttosql():
    try:
        cnxn1 = psycopg2.connect(user="postgres",password="abcd@1234",host="192.168.1.12",port="5432",database="p66DB1")
        cursor1 = cnxn1.cursor()
        
        if (cnxn1):
            pass
            #logData("Connection to AK PC server successful")
        else:
            logData("Connection to AK PC server unsuccessful")
        return(cnxn1,cursor1)    
    except:
        logData('problem in connecting to sql')
        
        
def update_data_1_min(max_time_input_data,cnxn1):
        q3="select * from pull_raw_data_1_min (t1:='"+str(max_time_input_data)+"');"
        df=pd.read_sql(q3,cnxn1)
        df=df[['Time','Tag','Value']]
        df['Time']=max_time_input_data
        engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
        df.to_sql('input_data_1_min',engine,if_exists='append',index=False)
        logData('Updating in input_data_1_min,Time stamp value for '+str(max_time_input_data))
def run_aggregator_1_min():
    try:
        cnxn1,cursor1=connecttosql()
        cursor1.execute('select max("Time") from input_data_1_min limit 1')
        max_time_input_data=cursor1.fetchall()[0][0]
    
        cursor1.execute('select max("Time") from raw_data limit 1')
        max_time_raw_data=cursor1.fetchall()[0][0]
        
        if max_time_input_data is None:
            cursor1.execute('select datetime from filelist order by datetime limit 1;')
            min_time_filelist=cursor1.fetchall()[0][0]
            min_time_filelist=min_time_filelist.replace(second=0)
            max_time_input_data=min_time_filelist
            update_data_1_min(max_time_input_data,cnxn1)
    
        elif max_time_raw_data>max_time_input_data:
            max_time_input_data=max_time_input_data+timedelta(minutes=1)
            update_data_1_min(max_time_input_data,cnxn1)
           
        elif max_time_raw_data <= max_time_input_data:
            pass
            #logData('No new data available')
            
    except:
        logData('Some problem in updating data') 
        
      
def update_data_5_min(df_5_min,cnxn1):
        engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
        df_5_min.to_sql('input_data_5_min',engine,if_exists='append',index=False)
        logData('Updating in input_data_5_min,Time stamp value for '+str(df_5_min['Time'].values[0]))
    
def calculate_average(t1,cnxn1):
    df_data=pd.DataFrame()
    date_list = [t1 + timedelta(minutes=1*x) for x in range(0, 5)]
    for i in date_list:
        #print(i)
        q="""select * from input_data_1_min where input_data_1_min."Time" = '"""+str(i)+"""'"""
        df=pd.read_sql(q,cnxn1)
        df=df[['Tag','Value']]
        df['Value']=df['Value'].apply(pd.to_numeric) 
        df=df.set_index("Tag")
        df=df.T
        new_index=[i]
        df.insert(0, 'Time', i)
        df=df.reset_index(drop=True)
        df_data=df_data.append(df, ignore_index=True)
    mean_df=df_data[1:].mean(axis=0)        
    df_5_min=pd.DataFrame(mean_df)
    df_5_min['Tag']=df_5_min.index
    tt=t1+timedelta(minutes=4)
    df_5_min=df_5_min.rename(columns={0:'Value'})
    df_5_min['Time']=tt
    df_5_min=df_5_min.reset_index(drop=True)
    df_5_min=df_5_min[['Time','Tag','Value']]
    update_data_5_min(df_5_min,cnxn1)
    
   
def run_aggregator_5_min():  
    try:
        cnxn1,cursor1=connecttosql()
        cursor1.execute('select max("Time") from input_data_1_min limit 1')
        max_time_input_data_1_min=cursor1.fetchall()[0][0]
        cursor1.execute('select max("Time") from input_data_5_min limit 1')
        max_time_input_data_5_min=cursor1.fetchall()[0][0]
        if max_time_input_data_5_min is None:
            cursor1.execute('select min("Time") from input_data_1_min;')
            tm=cursor1.fetchall()[0][0]
            tm = tm - timedelta(minutes=tm.minute % 5,seconds=tm.second,microseconds=tm.microsecond)
            t1=tm+timedelta(minutes=6)
            calculate_average(t1,cnxn1)
    
        elif max_time_input_data_1_min >= max_time_input_data_5_min+timedelta(minutes=5):
            cursor1.execute('select max("Time") from input_data_5_min;')
            tm=cursor1.fetchall()[0][0]
            t1=tm+timedelta(minutes=1)
            calculate_average(t1,cnxn1)
    
        elif max_time_input_data_1_min < max_time_input_data_5_min+timedelta(minutes=5):
            pass
            #logData('No new data available')
            
    except:
        logData('Problem in aggregator_5_min')


def run():
    for i in range(2000):
        run_aggregator_1_min()
        run_aggregator_5_min()
        
#run()
    
class PythonCornerExample(SMWinservice):    
    _svc_name_ = "aggregator_p66_service"
    _svc_display_name_ = "aggregator_p66_service"
    _svc_description_ = "This service pulls data from raw_data table and paste to 1_min avg,it also pulls data from 1_min table and paste to 5_min avg"

    def start(self):
        self.isrunning = True

    def stop(self):
        self.isrunning = False

    def main(self):
        run()

if __name__ == '__main__':
    PythonCornerExample.parse_command_line() 
    
        
        # df=df[['Tag','Value']]
        # df=df.set_index("Tag")
        # a=df.T
        # new_index=[max_time_input_data]
        # a=a.reset_index(drop=True)
        
        
        
        
    #     a.insert(0, 'Time', max_time_input_data)
    
    #     df2 = a.pivot(index='Tag', columns='Value')
    # # if max_time_raw_data > max_time_input_data:
    #     q3='select * from pull_raw_data_1_min (t1:='+str(max_time_input_data)');'
        
        
    # s=max_time_input_data
    # q='select max("Time") from input_data_1_min limit 1'
    # max_time_input_data=ti=cursor1.fetchall()[0][0]
    
        #df=pd.read_sql(postgreSQL_select_Query,cnxn1)
    # if df.shape[0]==0:
    #     cursor1.execute('select max("Time") from raw_data')
    #     ti=cursor1.fetchall()[0][0]
    #     print('a')
    #     if ti>tf:
            
    #         print('a')
    #     else:
    #         print('b')







        # q3="select * from pull_raw_data_1_min (t1:='"+str(max_time_input_data)+"');"
        # df=pd.read_sql(q3,cnxn1)
        # df=df[['Time','Tag','Value']]
        # df['Time']=max_time_input_data
        # engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
        # df.to_sql('input_data_1_min',engine,if_exists='append',index=False)
        # logData('Updating in input_data_1_min,Time stamp value for '+str(max_time_input_data))
        

        # q3="select * from pull_raw_data_1_min (t1:='"+str(max_time_input_data)+"');"
        # df=pd.read_sql(q3,cnxn1)
        # df=df[['Time','Tag','Value']]
        # df['Time']=max_time_input_data
        # engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
        # df.to_sql('input_data_1_min',engine,if_exists='append',index=False)
        # logData('Updating in input_data_1_min,Time stamp value for '+str(max_time_input_data))
        


    #raw_data_Tags='"'+str('","'.join(raw_data_Tags['Tag'].tolist()))+'"'
    #'''SELECT DISTINCT ON ("Tag") "Tag","Time","Value" FROM raw_data WHERE "Tag" in ('''+str(raw_data_Tags)''') AND "Time" < '2020-12-3 08:28:00' ORDER BY "Tag","Time" DESC;'''
    
    # engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
    # raw_data_Tags.to_sql('raw_data_tags',engine,if_exists='replace',index=False)
    #raw_data_Tags='"'+str('","'.join(raw_data_Tags['Tag'].tolist()))+'"'

