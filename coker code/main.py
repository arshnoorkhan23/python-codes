
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

folderName = os.path.dirname(__file__)
all_mails_file_path=os.path.join(folderName,'all_mails_files')
if not os.path.exists(all_mails_file_path):
    os.makedirs(all_mails_file_path)
    print('making all_mails_file_path directory')
    
logging.basicConfig(filename=os.path.join(folderName,'log_file.log'), filemode='a', level=logging.INFO ,format='%(asctime)s: %(message)s', datefmt='%d %b %y %H:%M:%S')
def logData(statusText):
    currTime = (datetime.now())
    print ("{} : {}".format(str(currTime)[:-7], statusText))
    logging.info(statusText)

#----------------------------------------Connecting the created database-------------
    
def connecttosql():
    try:
        cnxn1 = psycopg2.connect(user="postgres",
                                  password="abcd@1234",
                                  host="192.168.1.12",
                                  port="5432",
                                  database="p66DB1")
        cursor1 = cnxn1.cursor()
        
        if (cnxn1):
            logData("Connection to AK PC server successful")
        else:
            logData("Connection to AK PC server unsuccessful")
        return(cnxn1,cursor1)    
    except:
        logData('problem in connecting to sql')
    
cnxn1,cursor1=connecttosql()

def refine_df(df_data):
    df_data['Value'].replace([np.inf, -np.inf], np.nan, inplace = True)
    df_data['Value'] = df_data['Value'].fillna(0)
    df_data.dropna(subset = ["Tag"], inplace=True)
    df_data.dropna(subset = ["Time"], inplace=True)
    df_data=df_data[df_data['Time'] !='Tag not found']

    return(df_data)

iteration=0

while 1:
    iteration=iteration+1
    logData('iteration number '+ str(iteration))
    username = "sbhatt@ingenero.com"
    password = "Ing@2021"
    
    # create an IMAP4 class with SSL 
    imap = imaplib.IMAP4_SSL("imap.ingenero.com")
    # authenticate
    imap.login(username, password)
    
    status, messages = imap.select("INBOX")
    # number of top emails to fetch
    if iteration >1:
        N=10
    else:
        N=100
    
    # total number of emails
    messages = int(messages[0])
    
    for i in range(messages, messages-N, -1):
        # fetch the email message by ID
        res, msg = imap.fetch(str(i), "(RFC822)")
        for response in msg:
            if isinstance(response, tuple):
                # parse a bytes email into a message object
                msg = email.message_from_bytes(response[1])
                # decode the email subject
                subject = decode_header(msg["Subject"])[0][0]
                if isinstance(subject, bytes):
                    # if it's a bytes, decode to str
                    subject = subject.decode()
                # decode email sender
                From, encoding = decode_header(msg.get("From"))[0]
                if isinstance(From, bytes):
                    From = From.decode(encoding)
                print("Subject:", subject)
                print("From:", From)
                # if the email message is multipart
                if msg.is_multipart():
                    # iterate over email parts
                    for part in msg.walk():
                        # extract content type of email
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        try:
                            # get the email body
                            body = part.get_payload(decode=True).decode()
                        except:
                            pass
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            # print text/plain emails and skip attachments
                            print(body)
                        elif "attachment" in content_disposition:
                            # download attachment
                            filename = part.get_filename()
                            if filename:
                                #if not os.path.isdir(subject):
                                #if not os.path.exists(os.path.join(all_mails_file_path,subject)):
                                if not os.path.exists(os.path.join(all_mails_file_path,filename)):
                                    ab=os.path.join(all_mails_file_path,subject)
                                    filepath=os.path.join(all_mails_file_path,filename)
                                    open(filepath, "wb").write(part.get_payload(decode=True))
                                    logData('Saving file:- -'+filename)

                else:
                    # extract content type of email
                    content_type = msg.get_content_type()
                    # get the email body
                    body = msg.get_payload(decode=True).decode()
                    if content_type == "text/plain":
                        # print only text email parts
                        print(body)
                if content_type == "text/html":
                    # if it's HTML, create a new HTML file and open it in browser
                    if not os.path.isdir(subject):
                        # make a folder for this email (named after the subject)
                        os.mkdir(subject)
                    filename = f"{subject[:50]}.html"
                    filepath = os.path.join(subject, filename)
                    # write the file
                    open(filepath, "w").write(body)
                    # open in the default browser
                    webbrowser.open(filepath)
                print("="*100)
                
    # close the connection and logout
    imap.close()
    imap.logout()    

    #---------------------------Listing all downloaded files----------------
    df_info=pd.DataFrame()                                                 
    all_mails_folder_list = [ item for item in os.listdir(all_mails_file_path)]
    df_info['name_1']=all_mails_folder_list
    
    
    time_of_file=[item[-19:-17]+'-'+item[-17:-15]+'-'+item[-15:-11]+' '+item[-10:-8]+':'+item[-8:-6]+':'+item[-6:-4] for item in df_info['name_1']]
    time_formatted=[datetime.strptime(item,'%d-%m-%Y %H:%M:%S') for item in time_of_file]
    df_info['datetime']=time_formatted
       
    df_info = df_info.sort_values(by="datetime")
    df_info_all=df_info

    #---------------------------Listing all downloaded files----------------

    #-----------------------last updated time stamp of files 1592--------------------
    try:
        postgreSQL_select_Query = "select * from filelist"
        df_filelist=pd.read_sql(postgreSQL_select_Query,cnxn1)
        df_info_to_be_updated= (pd.merge(df_info,df_filelist,indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1))

    except:
        logData('An error observed while pulling last value of tag XSD1405 which is of file 1592')
    
    for i,file in df_info_to_be_updated.iterrows():
        try:
            df_data=pd.read_csv(os.path.join(all_mails_file_path,file.name_1),header=None,names=['Tag','Time','Value','Time_updated'])
            df_data=df_data[['Time','Tag','Value']]
            df_data=refine_df(df_data)
            df_data['Time'] = pd.to_datetime(df_data['Time'],format='%d-%b-%y %H:%M:%S')
            engine = create_engine('postgresql+psycopg2://postgres:abcd@1234@192.168.1.12:5432/p66DB1')
            df_data.to_sql('raw_data',engine,if_exists='append',index=False)
            df_info_to_be_updated.loc[i:i,:].to_sql('filelist',engine,if_exists='append',index=False)
            logData('Updated in postgreSQL file:- '+str(file))
        except:
            logData('Problem in Updating in postgreSQL server file '+str(file))
            
    #-----------------Uploading data of 1592 files------------------------
    
    print('done')
    print('do')
    
    


