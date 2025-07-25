import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preparation for UT Plots
def data_preparation_for_ut(data):
    #Considering only one plane
    data_ = data[data.Sector.str.startswith('UTaX')][['Staves','Rows','PedestalValue']]


    data_ = data_.groupby(['Staves','Rows']).agg('mean').reset_index()
    data_.head()

    #Providing the actual order of the UT-Detector
    row_order = ['S4T','M4T','S3T','M3T','S2T','M2T','S1T','M1T','M1B','S1B','M2B','S2B','M3B','S3B','M4B','S4B']
    stave_order = ['1C', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '8A', '7A', '6A', '5A', '4A', '3A', '2A', '1A']

    data_['Rows'] = pd.Categorical(data_['Rows'], categories=row_order, ordered=True)
    data_['Staves'] = pd.Categorical(data_['Staves'], categories=stave_order, ordered=True)


    data_ = data_.pivot(index='Rows', columns='Staves', values='PedestalValue')
    return data_
