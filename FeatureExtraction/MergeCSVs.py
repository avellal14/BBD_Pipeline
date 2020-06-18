"""
MergeCSVs.py

This file contains the function which combines all the feature .csv files from each individual
WSI into a single CSV where each column contains all the feature values for a single patient.
"""


import os
import time
import pandas as pd
import numpy as np
import csv

############################################################################################
def merge_csvs(dir_nm, feature_list_csv, master_csv_nm):
    """
    merge_csvs takes an input dir containing feature .csvs for individual WSIs, then
    aggregates them into a single .csv (only keeps the features specified in feature_list_csv), 
    saved at the directory master_csv_nm, where each column is headed by a patient's ID 
    and contains all the feature values for that patient

    param: dir_nm, feature_list_csv, master_csv_nm
    return: aggregate .csv file with patient-level feature values saved at master_csv_nm
    """


    t = time.time() #keep track of the starting time

    #obtain list of features to save for each patient
    feature_list = []
    with open(feature_list_csv, 'rb') as f:
    	reader = csv.reader(f)
        feature_list = list(reader)
        
    feature_list = [feature[0] for feature in feature_list]

    #obtain all the individual feature .csv files in the directory
    file_list = [name for name in os.listdir(dir_nm) if os.path.isfile(os.path.join(dir_nm, name))]
    file_list = [file_nm for file_nm in file_list if 'csv' in file_nm]
    file_list.sort()

    col_list = list() #create an empty column list

    #initialize the master dataframe by reading in the first feature .csv in the list
    master_DF = pd.read_csv(os.path.join(dir_nm, file_list[0]), names=['Name'],dtype={'Val':np.float32}).T
    master_DF = pd.read_csv(feature_list_csv,names=['Name'])

  
    #loop through all the files and obtain all the patient IDs
    for filenm in file_list:
        positions = [pos for pos, char in enumerate(os.path.basename(filenm)) if char == '_']
        colnm = os.path.basename(filenm)[:positions[0]]
        if (colnm not in col_list): col_list.append(colnm)

    #iterate through all the patients
    for current_patient in col_list:
        patient_WSInm_list = [name for name in file_list if current_patient in name]
        patient_WSI_data = np.zeros([len(feature_list),1],dtype=np.float32) #make a list of dfs storing the info of all patients

        #average all the feature data over all WSIs belonging to the current patient
        for wsi in patient_WSInm_list:
            #current_DF = pd.read_csv(os.path.join(dir_nm,wsi),names=["Name", "Val"],dtype={"Val":np.float32},usecols=["Name", "Val"]).T
            
            current_DF = pd.read_csv(os.path.join(dir_nm,wsi),names=['Val'], dtype={'Val':np.float32}).T 
            current_DF = current_DF[feature_list].T
	    patient_WSI_data = patient_WSI_data + current_DF.as_matrix()

        patient_WSI_data = patient_WSI_data/len(patient_WSInm_list)
        master_DF[current_patient] = patient_WSI_data #place the patient-level feature data in one column of the master feature dataframe
               

    master_DF.to_csv(master_csv_nm) #save the master feature dataframe in a .csv file at the location specified by master_csv_nm
    print("Time Elapsed for Aggregating Features: " + str(time.time() - t)) #print the total time elapsed for aggregating all the feature .csv files


#TODO: Jan, please change the line below to point to the directory corresponding to all of the BBD images 
merge_csvs(os.path.join('/n', 'groups', 'becklab', 'EpiStroma', 'BBDNCC_features'),
           os.path.join('/n', 'groups', 'becklab', 'EpiStroma', 'FeatureExtractionCode_AV_KS-20181130', 'FeatureExtractionCode_AV_KS', 'feature_list.csv'),
           os.path.join('/n', 'groups', 'becklab', 'EpiStroma', 'FeatureExtractionCode_AV_KS-20181130', 'FeatureExtractionCode_AV_KS', 'all_features.csv'))

#merge_csvs(os.path.join('R:', os.sep, 'BBD_Images'))
