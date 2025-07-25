import pandas as pd

def dataBuilder(run_number):

    data_directory_path = '/home/hashmi/FileDrive/UpstreamTrackerCalibration/UTCOND/NZS_data'
    
    # Importing the datasets for pre-processing
    CMS_noise = pd.read_csv(f'{data_directory_path}/{run_number}/CMS_noise.csv',names=['ChannelID','Signal','CMSubstracted','Int']).drop(['Int'],axis=1)

    commonMode = pd.read_csv(f'{data_directory_path}/{run_number}/commonMode.csv',names=['ChipID','ChipMean','ChipSigma','Int']).drop(['Int'],axis=1)

    pedestals = pd.read_csv(f'{data_directory_path}/{run_number}/pedestals.csv',names=['ChannelID','PedestalValue',])

    sigmaNoise = pd.read_csv(f'{data_directory_path}/{run_number}/sigmaNoise.csv',names=['ChannelID','Sigma','STD'])


    #Importing Pickle Files for Mapping
    translator_=pd.read_pickle('/home/hashmi/FileDrive/UpstreamTrackerCalibration/NewNotebooks/Data/translator.pkl')
    universal_map_ = pd.read_pickle('/home/hashmi/FileDrive/UpstreamTrackerCalibration/NewNotebooks/Data/universal_map.pkl')

    #Converting them to datasets for processing
    translator = pd.DataFrame.from_dict(translator_,orient='index').reset_index()
    
    translator.columns=['ChannelID','ChipID']

    translator[['Sector','ChipID','ChannelNumber']]=translator['ChipID'].str.rsplit('.',n=2,expand=True)

    # Processing Universel Map
    universal_map=pd.DataFrame.from_dict(universal_map_,orient='index').reset_index()[['index','sensor_type']]
    universal_map.columns=['ChipID','SensorType']
    universal_map[['Sector','ChipID']]=universal_map['ChipID'].str.rsplit('.',n=1,expand=True)



    
    #Final Preprocessing of imported datasets
    commonMode[['Sector','ChipID']]=commonMode['ChipID'].str.rsplit('_', n=1, expand=True)

    commonMode['ChipID']=commonMode['ChipID'].map({'0':'Chip0','1':'Chip1','2':'Chip2','3':'Chip3'})

    # Preparing the New Datasets combining All the Above:
    data = translator.merge(CMS_noise,how='left')

    data = data.merge(commonMode,on=['Sector','ChipID'],how='left')

    data = data.merge(pedestals,on='ChannelID',how='left')

    data = data.merge(sigmaNoise,on='ChannelID',how='left')

    data = data.merge(universal_map,on=['Sector','ChipID'],how='left')


    # Preparing Staves and Row Informations:

    #ECS has a naming structure, [UTaX_1AB_M1W_0]
    #Split them based on '_'
    sector_ =  data.Sector.str.split('_',expand=True)

    #Taking the second part of the name and picking 1A that identifies the stave position and the side [A/C]
    staves = sector_[1].str[:2]

    #Section defines the section if it is Top[T] or Bottom[B] of the plane
    section = sector_[1].str[2:]

    #Taking the Row details from the third piece of ECS name, ignoring [E/W] for the moment
    rows_ = sector_[2].str[:2]

    #Combining the Row details and adding it with the section half identifier producing similar to [M4+T]
    rows = rows_+section

    
    #Adding Staves and Rows data to the table before merging.
    data[['Staves', 'Rows']] = pd.DataFrame({'Staves': staves, 'Rows': rows})


    data[['Signal','CMSubstracted','ChipMean','ChipSigma','PedestalValue']] = data[['Signal','CMSubstracted','ChipMean','ChipSigma','PedestalValue']].fillna(0)


    #For Run Number Identifier
    data['RunNumber'] = int(run_number)

    return data

    
