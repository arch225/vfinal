import streamlit as st
import pandas as pd
import pickle
import sqlite3
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer,GaussianCopulaSynthesizer,TVAESynthesizer
from sdv.lite import SingleTablePreset
from datetime import date

##oh yeaaaaahh
def Gen_data(model,rows):
    try:
        if model=='FastML':
            m_obj='my_synth_fml_v1.pkl'
        elif model=='Gaussian Copula Synthesizer':
            m_obj='my_synth_gcs_v1.pkl'
        elif model=='CT-GAN':
            m_obj='my_synth_ctg_v1.pkl'
        elif model=='TVAE':
            m_obj='my_synth_tvae_v1.pkl'
        with open(m_obj,'rb') as f:
            
            syn=pickle.load(f)
            row=int(rows)
            
            syn_gen_data=syn.sample(num_rows=row)
            return syn_gen_data
    except ValueError:
        return "Invalid option chosen."
    
def db_tables(db):
    conn = sqlite3.connect(db)
    sql_query = """SELECT name FROM sqlite_master  
    WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    df_address=pd.read_sql_query("SELECT * FROM ing_address",conn)
    df_customer=pd.read_sql_query("SELECT * FROM ing_customer",conn)
    df_demographic=pd.read_sql_query("SELECT * FROM ing_demographic",conn)
    df_termination=pd.read_sql_query("SELECT * FROM ing_termination",conn)
    df_mer1=pd.merge(df_customer,df_address,on='ADDRESS_ID',how='left')
    df_mer2=pd.merge(df_mer1,df_demographic,on='INDIVIDUAL_ID',how='left')
    df_mer3=pd.merge(df_mer2,df_termination,on='INDIVIDUAL_ID',how='left')
    st.write(df_mer3.head(10))
    return df_mer3

def show_tables(db):
    conn = sqlite3.connect(db)
    sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    cursor = conn.cursor()
    cursor.execute(sql_query)
    return (st.write(cursor.fetchall()))

def impute_all():
    st.write('Starting Imputation..')
    df2=df_mer3[df_mer3['INDIVIDUAL_ID'].notna()]
    df2['ACCT_SUSPD_DATE']=np.where(df2['ACCT_SUSPD_DATE'].isnull(),'2025-01-01',df2['ACCT_SUSPD_DATE'])
    d=[]
    for i in df2.columns:
        if df2[i].isna().sum()>0:
            d.append(i)
    else:
        print("Looks good: ",i,"\s",df2[i].isna().sum())
    for i in d:
        if (df2[i].dtype == 'int64') or (df2[i].dtype == 'float64'):
            df2[i]=df2[i].fillna(df2[i].median())
        elif df2[i].dtype == 'object':
            df2[i]=df2[i].fillna(df2[i].mode()[0])
    st.write("Imputation status",df2.isna().sum())
    return df2

def meta_validation(dat):
    st.write('Validating Metadata...')
    metadata=SingleTableMetadata()
    metadata.detect_from_dataframe(dat)
    py_dict=metadata.to_dict()
    st.write(py_dict)
    st.write('Validation completed')
    return metadata

def train_fml(dat,metadata):
    st.write('Training starting...')
    synth = SingleTablePreset(metadata, name='FAST_ML')
    synth.fit(df2)
    x='./custom_model/fml_'+date.today().strftime("%b-%d-%Y")+".pkl"
    synth.save(filepath=x)
    st.write(f'Training completed, model is saved: {x}')

def train_gcs(dat,metadata):
    st.write('Training starting...')
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df2)
    x='./custom_model/gcs_'+date.today().strftime("%b-%d-%Y")+".pkl"
    synth.save(filepath=x)
    st.write(f'Training completed, model is saved: {x}')

def train_ctgan(dat,metadata):
    st.write('Training starting...')
    synth = CTGANSynthesizer(metadata,epochs=100,verbose=True)
    synth.fit(df2)
    x='./custom_model/ctgan_'+date.today().strftime("%b-%d-%Y")+".pkl"
    synth.save(filepath=x)
    st.write(f'Training completed, model is saved: {x}')

def train_tvae(dat,metadata):
    st.write('Training starting...')
    synth = TVAESynthesizer(metadata,epochs=100,verbose=True)
    synth.fit(df2)
    x='./custom_model/tvae_'+date.today().strftime("%b-%d-%Y")+".pkl"
    synth.save(filepath=x)
    st.write(f'Training completed, model is saved: {x}')


st.title('Welcome to V1 version of data synthesizer :smile:')
task=st.sidebar.selectbox('Select your choice:',['Train a model','Generate data','Model Quality report'])            

st.divider()
if task=='Train a model':
    st.write('Connect to data source: Database or Flatfiles')
    dat_src=st.radio('Choose data source',['Connect to a db','Upload files'])
    st.divider()
    if dat_src=='Connect to a db':
        show_table_ind=st.checkbox('Show data in DB')
        if show_table_ind:
            st.write('Connecting to db...')
            show_tables('Source_data.db')
            load_data_ind=st.checkbox('Load data')
            if load_data_ind:
                df_mer3=db_tables('Source_data.db')
                st.write('Data has been loaded')
                missing_status_ind=st.checkbox('Show missing values in data')
                if missing_status_ind:
                    st.write(df_mer3.isna().sum())
                    impute_all_ind=st.checkbox('Impute all')
                    if impute_all_ind:
                        df2=impute_all()
                        meta_validation_ind=st.checkbox('Validate metadata')
                        if meta_validation_ind:
                            metadata=meta_validation(df2)
                            st.write('Train model')
                            col1,col2=st.columns(2)
                            with col1:
                                model_train=['FastML','Gaussian Copula Synthesizer','CT-GAN','TVAE']
                                mchoice=st.selectbox('Select model',model_train)
                            with col2:
                                epoch_input=st.number_input('Select Epochs:',1,1000,step=2)
                            train_ind=st.checkbox('Train model')
                            if train_ind:
                                if mchoice=='FastML':
                                    train_fml(df2,metadata)
                                elif mchoice=='Gaussian Copula Synthesizer':
                                    train_gcs(df2,metadata)
                                elif mchoice=='CT-GAN':
                                    train_ctgan(df2,metadata)
                                elif mchoice=='TVAE':
                                    train_tvae((df2,metadata))
elif task=='Generate data':
    models=['FastML','Gaussian Copula Synthesizer','CT-GAN','TVAE','Custom model']
    m_selection=st.selectbox('Choose a model to generate data: ',models,index=0)
    rows=st.number_input('Number of rows to be generated: ',min_value=1,max_value=1000000,value=1000)
    st.write(f'So you have chosen model: {m_selection} and {rows} number of rows to be generated')
    gen_button=st.button('Generate Data')
    st.divider()
    if gen_button:
        st.write('Generating data....Please wait..')
        df_synth=Gen_data(m_selection,rows)
        st.write(df_synth)
        
        l_add=['ADDRESS_ID', 'LATITUDE', 'LONGITUDE', 'STREET_ADDRESS', 'CITY','STATE', 'COUNTY']
        l_customer=['INDIVIDUAL_ID', 'ADDRESS_ID', 'CURR_ANN_AMT', 'DAYS_TENURE','CUST_ORIG_DATE', 'AGE_IN_YEARS', 'DATE_OF_BIRTH','SOCIAL_SECURITY_NUMBER'] 
        l_demographic=['INDIVIDUAL_ID', 'INCOME', 'HAS_CHILDREN', 'LENGTH_OF_RESIDENCE','MARITAL_STATUS', 'HOME_MARKET_VALUE', 'HOME_OWNER', 'COLLEGE_DEGREE','GOOD_CREDIT']
        l_termination=['INDIVIDUAL_ID', 'ACCT_SUSPD_DATE']
        df_temp=df_synth[l_add]
        df_sys_add=df_temp.drop_duplicates(subset=['ADDRESS_ID'],keep='last')

        df_temp=df_synth[l_customer]
        df_sys_cust=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')

        df_temp=df_synth[l_demographic]
        df_sys_demo=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')

        df_temp=df_synth[l_termination]
        df_sys_term=df_temp.drop_duplicates(subset=['INDIVIDUAL_ID'],keep='last')

        one,two,three,four=st.columns(4)
        with one:
           st.text('Address file')
           st.write(df_sys_add)
        with two:
            st.text('Customer file')
            st.write(df_sys_cust)
        with three:
            st.text('Demographics file')
            st.write(df_sys_demo)
        with four:
            st.text('Termination file')
            st.write(df_sys_term)
elif task=='Model Quality report':
    st.write('Area under construction')