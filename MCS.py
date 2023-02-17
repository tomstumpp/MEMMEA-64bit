import numpy as np
import pandas as pd


def raw_import(file_name):
    print ('Opening file')
    f = open(file_name, 'rb')
    f.readline()
    f.readline()
    f.readline()
    sampling_rate = float(f.readline()[14:-2])
    print ('Sampling rate: '+str(sampling_rate))
    ADC_zero = float(f.readline()[11:-2])
    print ('Offset: '+str(ADC_zero))
    El = float(f.readline()[5:11])
    print ('Scaling: '+str(El))
    channel_names = str(f.readline()[10:]).split(';')
    print (str(len(channel_names))+' channels')
    channel_names[-1] = channel_names[-1][:-2]
    f.readline()

    from numpy import fromfile

    data = fromfile(f, dtype='uint16')
    data = data.reshape((len(channel_names),-1), order='F').astype(float)
    for i in range(len(channel_names)):
        if channel_names[i].endswith('\\r\\'):
            channel_names[i]=channel_names[i][:-3]
        if channel_names[i].startswith('b'):
            channel_names[i] = channel_names[i][2:]
        if channel_names[i].startswith('El'):
            data[i]=(data[i]-ADC_zero)*El
    f.close()
    return data, sampling_rate, channel_names

#file_name='I:\\BiomedMat\\1050_BMNT\\Team\\Stumpp\\MEMMEA\\Data\\Test_data\\From_Haein\\2022-11-25\\Nov3_22699_20221125_6wellorg_10p5_PTZ1.raw'
#input=raw_import(file_name)
#print('Status')