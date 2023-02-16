import pandas
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import poisson
import math
from matplotlib import pyplot as plt
import os
import MCS

np.random.seed(19680801)
rng=np.random.default_rng(12345)

def convert_input(sig):
    data,sampling_rate,column_names=MCS.raw_import(sig)
    sig_data=pd.DataFrame()
    data_time=[]
    name=np.split(sig,'/')[-1:]
    n=0
    for names in column_names:
        sig_data.assign(names=data[n,:])
        n += 1
    time_steps=1/sampling_rate
    for time_points in range(0,len(data[1,:]+1)):
        data_time.append(time_points*time_steps)
    sig_data.assign(Time=data_time)# clearing of dataframe needed!!! research how to!!!!
    burst=name.find("PTZ")
    return name,sig_data,burst

def sig_const(set):
    # the time signal should always be generated for 10s by default
    length=math.ceil(set.duration/set.time_steps)
    repeat_length=1
    signals=pd.DataFrame()
    signals = signals.assign(Time=np.arange(0, set.duration, set.time_steps))
    if set.execute_random==1 or True:
        signals=signals.assign(Random_Signal= rng.integers(2,size=length))
    if set.execute_evenly == 1 or True:
        array=[]
        for n in range(0,int(length/2)):
            array.append(1)
            for i in range(0,repeat_length):
                array.append(0)
        signals=signals.assign(Even_Spaced_Signal=array)
    if set.execute_gausian==True or 1:
        n,p= length,0.5
        x=np.arange(binom.ppf(0.01,n,p),binom.ppf(0.99,n,p))
        signals=signals.assign(Gausian_Signal= np.fft.ifft(a=binom.pmf(x, n, p),n=length).real*10**5)
        signals=signals.assign(Gausian_Signal=signals['Gausian_Signal'].values.astype(int))
        signals=signals.assign(Gausian_Signal =[1 if np.abs(signals['Gausian_Signal'].values[n])>=
                                                     np.mean(signals['Gausian_Signal'].values) else 0 for n in range(0,len(signals['Gausian_Signal'].values))])
    if set.execute_poison== True or 1:
        mu=0.6
        x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
        signals=signals.assign(Poisson_Signal=np.fft.ifft(a=poisson.pmf(x,mu),n=length).real*10**5)
        signals=signals.assign(Poisson_Signal=signals['Poisson_Signal'].values.astype(int))
        signals=signals.assign(Poisson_Signal=[1 if np.abs(signals['Poisson_Signal'].values[n])>=
                                                    np.mean(signals['Poisson_Signal'].values) else 0 for n in range(0,len(signals['Poisson_Signal'].values))])
    random_array=[]
    if set.amount_burst>2:
        zero_array=np.zeros(set.max_bursts-set.amount_burst)
        ones_array=np.ones(set.amount_burst)
        even_array=np.tile(np.concatenate((zero_array,ones_array)),int(length/len(zero_array/2)))
        if len(even_array)<=length:
            even_array=even_array.append(np.zeros(length-len(even_array)))
        else:
            even_array=even_array[:length]
        while len(random_array)<length:
            zero_array=np.zeros(np.random.randint(set.max_bursts))
            random_array=np.concatenate((random_array,zero_array,ones_array))  #continue here...
        random_array=random_array[:length]
        signals=signals.assign(Even_Burst=even_array)
        signals=signals.assign(Random_Burst=random_array)
    return signals

def read_memristor(sig,set,name):
    len_time=len(sig['Time'])
    read=pd.DataFrame(index=np.arange(0,len_time))
    font = {'size': 12,
            'weight': 4,
            'color': 'black',
            'verticalalignment': 'top',
            'horizontalalignment': 'center'}
    fig, axs = plt.subplots(1,(set.read_times_max+1-set.read_times_min))
    figstep, axsstep=plt.subplots((set.read_times_max+1-set.read_times_min),1)
    for n in np.arange(set.read_times_min,set.read_times_max+1):
        min=set.read_times_min
        width = 0.04
        x_ticks = np.arange(1,n+1)
        axs[(n-min) - 2].set_xticks(x_ticks+width*n)
        axs[(n-min) - 2].set_ylabel('Counted_Spikes')
        axs[(n-min) - 2].set_xlabel('Time')
        for column_type in sig.columns[1:]:
            devision_point=math.ceil(len_time / n)
            read_times = []
            read_results = []
            for time_sections in np.arange(1,n+1):
                time=devision_point*time_sections
                read_results.append(np.sum(sig[column_type].values[0:time + 1])) # values are called which are not even in the array in some cases but the correct results is generated without error
                if time>100:
                    read_times.append(100)
                else:
                    read_times.append(time)
            read['Time_for_' + str(n)] = pd.Series(read_times)
            read[str(column_type)+'_measurment_'+str(n)]=pd.Series(read_results)
            axs[(n-min) - 2].bar(x=x_ticks+width*sig.columns.get_loc(column_type), height=read_results, width=width ,label=column_type)
            axsstep[(n - min) - 2].step(read_results,read_times,where='post')
        #read_results=read.iloc[:len(read_times), ((n - 2) * (len(sig.columns[1:]) + 1)):]
        #axs[n-2].plot(read_times,read_results)
        axs[(n-min) - 2].legend()
        axs[(n-min) - 2].set_xticklabels(read_times)
        axsstep[(n-min) - 2].legend()
        axsstep[(n - min) - 2].set_xticklabels(read_times)
    fig.suptitle('duration_' + str(set.duration) + '_time_steps_' + str(set.time_steps) + '_read_times_' + str(
        set.read_times_max - set.read_times_min) + '_spacing_' + str(n), fontdict=font)
    type='_bar_'
    fig.show()
    saveplot(name,read,set,fig,type)
    figstep.subtitle('duration_' + str(set.duration) + '_time_steps_' + str(set.time_steps) + '_read_times_' + str(
        set.read_times_max - set.read_times_min) + '_spacing_' + str(n), fontdict=font)
    type='_step_'
    figstep.show()
    saveplot(name,read,set,figstep,type)
    return read

def saveplot(name,read,set,figure,type):
    current_filepath = os.path.dirname(os.path.realpath(__file__))
    if set.save=='Y' or 'y':  #should be extra function
        if not name:
            filedirectory = os.path.join(current_filepath, "Output_Data/",
                                          'duration_' + str(set.duration) + '_time_steps_' + str(set.time_steps) + '_read_times_' + str(
                                         set.read_times_max - set.read_times_min)+ type+ '.png')
        else:
            filedirectory = os.path.join(current_filepath,name+type+'.png')
        figure.savefig(filedirectory)
        filedirectory = os.path.join(filedirectory[:-3] + '.csv')
        read.to_csv(filedirectory, sep='\t', index=False)



