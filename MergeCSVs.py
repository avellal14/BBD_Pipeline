import time
import pandas as pd
import numpy as np
#INPUT: Directory with one CSV for each WSI
#OUTPUT: Each column has all the data for one patient, with their ID at the top of the column
def mergeCSVs(dir):
    t = time.time()
    fileList = getAllFiles(dir)
    fileList.sort()
    colList = list()
    masterCSVNm = os.path.join('/home','avellal14','data','features_all_white_space.csv')
    masterDF = pd.read_csv(os.path.join(dir,fileList[0]), names = ["Name", "Val"],usecols=["Name"])

    # loop through, get all patient IDs first
    for filenm in fileList:
        positions = [pos for pos, char in enumerate(os.path.basename(filenm)) if char == '_']
        colName = filenm[:-37] #TODO: obtain patient Name

        if(colName not in colList): colList.append(colName) #if patient has not been seen before, add patient to list

    #loop through all patients
    for currentPatient in colList:
        patientWSINmList = [name for name in fileList if currentPatient in name]
        patientWSIdata = np.zeros([264,1],dtype=np.float32) #TODO: originally 508 when considering all features - 264 for haralick and texture features. make a list of dfs storing the info of all patients

        #loop through all WSIs for patient
        for wsi in patientWSINmList:
            currentDF = pd.read_csv(os.path.join(dir,wsi),names=["Name", "Val"],dtype={"Val":np.float32},usecols=["Val"])
            patientWSIdata = patientWSIdata + currentDF.as_matrix() #sum the data from all WSIs belonging to the patient

        print("Current Patient: " + currentPatient + "  Num Files: " + str(len(patientWSINmList)))
        patientWSIdata = patientWSIdata/len(patientWSINmList) #take the mean of all features over the number of WSIs
        masterDF[currentPatient] = patientWSIdata #store the data for the patient in the dataframe

    masterDF.to_csv(masterCSVNm) #save the dataframe as a csv
    print("TIME: " + str(time.time() - t))


#INPUT: parent directory
#OUTPUT: all feature files in the directory
def getAllFiles(parentDir):
    return [name for name in os.listdir(parentDir)
            if os.path.isfile(os.path.join(parentDir, name)) and 'white' in name]

mergeCSVs(os.path.join('/data', 'avellal14', 'Adithya_BBD_NHS', 'BBD_NCC_extractedat20x_round2'))
