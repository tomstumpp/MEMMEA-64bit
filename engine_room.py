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
    name=sig.split('\\')[-1:]
    time_steps = 1 / sampling_rate
    for time_points in range(0,len(data[1,:]+1)):
        data_time.append(time_points*time_steps)
    sig_data['Time'] = data_time
    n=0
    for names in column_names:
        sig_data[names]=data[n,:]
        n+=1
    burst=bool(name[0].find("PTZ"))
    return name,sig_data ,burst, time_steps

def org_plot(name,sig,set):
    subplot_rows=math.ceil(len(sig.columns)/4)
    figstep, axsstep = plt.subplots(subplot_rows, 4)
    time=sig.iloc[:,0]
    for k in np.arange(0, subplot_rows):
        for n in np.arange(0,4):
            axsstep[k][n].set_ylabel('Counted_Spikes')
            axsstep[k][n].set_xlabel('Time')
            axsstep[k][n].step(sig.iloc[:,k*n+1], time, where='pre', label=str(sig.columns[k*n+1]))
    figstep.suptitle(name, fontdict=set.font)
    saveplot(name,sig,set,figstep,'step')

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

def write_memristor(sig,set):
    len_time=len(sig['Time'])
    channel_number=1
    column_amount=len(sig.columns)-1
    while channel_number <=column_amount:
        channel_data=np.array(sig.iloc[:,channel_number]*1000)
        threshold=set.threshold*1000
        channel_data_clone=np.concatenate((channel_data[1:],[0]))
        f_channel_data=channel_data > threshold
        f_channel_data_clone=channel_data_clone > threshold
        f_channel_data_clone=~f_channel_data_clone
        sig.iloc[:, channel_number]=pd.Series(np.array([x and y for x, y in zip(f_channel_data, f_channel_data_clone)]).astype(float))
        print('Conversion done:'+str(channel_number)+'_of_'+str(column_amount))
        channel_number+=1
    return sig

def read_memristor(sig,set,name,time_steps):
    len_time = len(sig['Time'])
    max_t_plot=10
    base_name= str(name)+'_syn_bursts_'+str(set.amount_bursts)+'_duration_' + str(set.duration) + '_time_steps_' + str(set.time_steps) + '_max_read_time_' + str(
        set.read_times_max)+ '_min-read_time_' + str(set.read_times_min) + '_spacing_'
    amount_plots = math.ceil((len_time * time_steps) / max_t_plot)
    if len_time*time_steps>=max_t_plot:
        previous_index=0
        for n in np.arange(1,amount_plots+1):
            if n==amount_plots:
                index_t_period = len_time-1
            else:
                index_t_period=sig.index[sig['Time']>=n*max_t_plot][0]
            name=base_name+'_'+str(n)+'separate_plot'
            section_plot(sig.iloc[previous_index:index_t_period,:],set,name)
            previous_index=index_t_period+1
    else:
        section_plot(sig,set,base_name)

def section_plot(sig,set,name):
    sig=sig.reset_index(drop=True)
    len_time=len(sig['Time'])-1
    time_array=sig['Time']
    read=pd.DataFrame(index=np.arange(0,len_time))
    fig, axs = plt.subplots(1,(set.read_times_max+1-set.read_times_min))
    figstep, axsstep=plt.subplots(1,(set.read_times_max+1-set.read_times_min))
    if (set.read_times_max+1-set.read_times_min)==1:
        subplot_amount=set.read_times_min
    else:
        subplot_amount=np.arange(set.read_times_min,set.read_times_max+1)
    s_number=0
    for n in subplot_amount:
        min=set.read_times_min
        width = 0.04
        x_ticks = np.arange(1,n+1)
        axs[s_number].set_xticks(x_ticks+width*n)
        axs[s_number].set_ylabel('Counted_Spikes')
        axs[s_number].set_xlabel('Time')
        axsstep[s_number].set_ylabel('Counted_Spikes')
        axsstep[s_number].set_xlabel('Time')
        read_times=[]
        devision_point = math.ceil(len_time / n)
        time_index=[]
        for time_sections in np.arange(1, n + 1):
            time_index_temp = devision_point * time_sections - 1
            time_index.append(time_index_temp)
            if time_index_temp >= len_time:
                read_times.append(time_array[len_time])
            else:
                read_times.append(time_array[time_index_temp])
        read['Time_for_' + str(n)] = pd.Series(read_times)
        for column_type in sig.columns[1:]:
            read_results = []
            for time_sections in np.arange(0,n):
                read_results.append(np.sum(sig[column_type].values[0:time_index[time_sections]+ 1])) # values are called which are not even in the array in some cases but the correct results is generated without error
            read[str(column_type)+'_measurment_'+str(n)]=pd.Series(read_results)
            axs[s_number].bar(x=x_ticks+width*sig.columns.get_loc(column_type), height=read_results, width=width ,label=column_type)
            axsstep[s_number].step(read_times,read_results,where='pre',label=column_type)
        #read_results=read.iloc[:len(read_times), ((n - 2) * (len(sig.columns[1:]) + 1)):]
        #axs[n-2].plot(read_times,read_results)
        axs[s_number].set_xticklabels(np.array(read_times).round(decimals=2))
        s_number +=1
    if n==subplot_amount.max():
        axs[s_number-1].legend()#bbox_to_anchor=(1.05, 1), loc='upper left')
        axsstep[s_number-1].legend()#bbox_to_anchor=(1.05, 1), loc='upper left')
    name=name+str(n)
    fig.suptitle(name, fontdict=set.font)
    type='_bar_'
    fig.show()
    saveplot(name,read,set,fig,type)
    figstep.suptitle(name, fontdict=set.font)
    type='_step_'
    figstep.show()
    saveplot(name,read,set,figstep,type)
    return read

def saveplot(name,read,set,figure,type):
    current_filepath = os.path.dirname(os.path.realpath(__file__))
    if set.save in 'Y' or 'y':  #should be extra function
        filedirectory = os.path.join(current_filepath,'Output_Data\\'+str(name)+str(type)+'.png')
        figure.savefig(filedirectory)
        filedirectory = os.path.join(filedirectory[:-3] + '.csv')
        read.to_csv(filedirectory, sep='\t', index=False)



