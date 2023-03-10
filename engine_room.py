import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import poisson
import math
from matplotlib import pyplot as plt
import os
import MCS
import matplotlib as mpl


np.random.seed(19680801)
rng=np.random.default_rng(12345)

def convert_input(sig): #reads in data from MCS .raw data
    data,sampling_rate,column_names=MCS.raw_import(sig)
    sig_data=pd.DataFrame()
    data_time=[]
    name=str(sig.split('/')[-1:][0])
    time_steps = 1 / sampling_rate
    for time_points in range(0,len(data[1,:]+1)):
        data_time.append(time_points*time_steps)
    sig_data['Time'] = data_time
    n=0
    for names in column_names:
        sig_data[names]=data[n,:]
        n+=1
    if bool(name[0].find("PTZ"))==1: 
        burst=True
    return name,sig_data ,burst, time_steps #name: (str) name of original data; sig_data: (pandas dataframe) with all data and time; burst: (bool) indicator if there might be spikes or not - not important right now but could be useful later, time_steps: (float) periodendauer 

def org_plot(name,sig,set): #currently not in use!!!
    subplot_rows=math.ceil(len(sig.columns)/4)
    figstep, axsstep = plt.subplots(subplot_rows, 4)
    time=sig.iloc[:,0]
    for k in np.arange(0, subplot_rows):
        for n in np.arange(0,4):
            axsstep[k][n].set_ylabel('Counted_Spikes')
            axsstep[k][n].set_xlabel('Time')
            axsstep[k][n].step(sig.iloc[:,k*n+1], time, where='pre', label=str(sig.columns[k*n+1]))
    figstep.suptitle(name, fontdict=set.font)
    saveplot(name,sig,set,figstep,'step') #constructed to plot a step function from the raw data

def sig_const(set): #creates all "synthetic" signals
    # the time signal should always be generated after the pre-set in the Crew file
    length=math.ceil(set.duration/set.time_steps)
    repeat_length=1 #distance between 0 and 1 @ evenly spaced signal
    signals=pd.DataFrame()
    signals = signals.assign(Time=np.arange(0, set.duration, set.time_steps))
    if set.execute_random==True:
        signals=signals.assign(Random_Sig= rng.integers(2,size=length))
    if set.execute_evenly == True:
        array=[]
        for n in range(0,int(length/2)):
            array.append(1)
            for i in range(0,repeat_length):
                array.append(0)
        signals=signals.assign(Even_Sp_Sig=array)
    if set.execute_gausian==True:
        n,p= length,0.5
        x=np.arange(binom.ppf(0.01,n,p),binom.ppf(0.99,n,p)) #binom since then its discrete and not continous - shouldn't make a difference
        signals=signals.assign(Gausian_Sig= np.fft.ifft(a=binom.pmf(x, n, p),n=length).real*10**5)#irfft
        signals=signals.assign(Gausian_Sig=signals['Gausian_Sig'].values.astype(int))
        signals=signals.assign(Gausian_Sig =[1 if np.abs(signals['Gausian_Sig'].values[n])>=
                                                     np.mean(signals['Gausian_Sig'].values) else 0 for n in range(0,len(signals['Gausian_Sig'].values))])
    if set.execute_poison== True:
        mu=0.6
        x = np.arange(poisson.ppf(0.01, mu), poisson.ppf(0.99, mu))
        signals=signals.assign(Poisson_Sig=np.fft.ifft(a=poisson.pmf(x,mu),n=length).real*10**5)#irfft
        signals=signals.assign(Poisson_Sig=signals['Poisson_Sig'].values.astype(int))
        signals=signals.assign(Poisson_Sig=[1 if np.abs(signals['Poisson_Sig'].values[n])>=
                                                    np.mean(signals['Poisson_Sig'].values) else 0 for n in range(0,len(signals['Poisson_Sig'].values))])
    if set.execute_burst==True:
        random_array=[]
        if set.amount_bursts>2:
            zero_array=np.zeros(set.max_bursts-set.amount_bursts)
            ones_array=np.ones(set.amount_bursts)
            even_array=np.tile(np.concatenate((zero_array,ones_array)),int(length/len(zero_array/2)))
            if len(even_array)<=length:
                even_array=even_array.append(np.zeros(length-len(even_array)))
            else:
                even_array=even_array[:length]
            while len(random_array)<length:
                zero_array=np.zeros(np.random.randint(set.max_bursts))
                random_array=np.concatenate((random_array,zero_array,ones_array))
            random_array=random_array[:length]
            signals=signals.assign(Even_Burst=even_array)
            signals=signals.assign(Random_Burst=random_array)
    return signals #returns synthetic signal dataframe with all "types" set to true

def write_memristor(sig,set):
    len_time=len(sig['Time'])
    channel_number=1
    converted_channel_number=1
    column_amount=len(sig.columns)-1
    sig_temp = pd.DataFrame(columns=['Time'])
    sig_temp['Time'] = sig['Time']
    # To determine which signal switched from above threshold to under threshold and therefore a "True" value is
    # is written into the result array, a second array is built "f_channel_data_clone" with minus one entry from the
    # original one at the start.
    while channel_number <=column_amount:
        channel_data=np.array(sig.iloc[:,channel_number]*1000)
        threshold_min=set.threshold_min*1000
        channel_data_clone=np.concatenate((channel_data[1:],[0]))
        f_channel_data=channel_data > threshold_min
        f_channel_data_clone=channel_data_clone > threshold_min
        f_channel_data_clone=~f_channel_data_clone
        converted_channel_data_temp=np.array([x and y for x, y in zip(f_channel_data, f_channel_data_clone)]).astype(float)
        if np.sum(converted_channel_data_temp)>5 and np.sum(converted_channel_data_temp)<500:#len_time/(len_time/1000):
            sig_temp[sig.columns[channel_number]]=pd.Series(converted_channel_data_temp)
            converted_channel_number +=1
            print('Conversion done:'+str(channel_number)+'_of_'+str(column_amount))
        channel_number+=1
        if channel_number==column_amount and converted_channel_number==1:
            sig_temp.assign(no_spikes=np.zeros((len_time,1)))
    selected_electrodes=str((input('The electrodes ' + str(list(sig_temp.columns[1:].values)) +
                   'were detected as having recorded spikes. '
                    'Which ones would you like to analyse further?(enter the number of the electrodes divided by a comma):')))
    if len(selected_electrodes)!=1:
        selected_electrodes=[int(x) for x in selected_electrodes.split(',')]
        selected_electrodes.insert(0, 0)
    else:
        selected_electrodes=[0,int(selected_electrodes)]
    sig_temp=sig_temp.iloc[:,selected_electrodes]
    return sig_temp #returns (dataframe) of all selected electrodes (starts @ 1) already thresholded - dataframe is only 0s and 1s 

def read_memristor(sig,set,name,time_steps): #function that creates plots from the sig data (from the write memristor function)
    len_time = len(sig['Time'])
    max_t_plot=100 #defines max s of a plot (just cut off to get individual plots from too long signals)
    synthetic_plot=str(int(set.execute_evenly))+str(int(set.execute_random))+str(int(set.execute_gausian))+str(int(set.execute_poison))
    if set.execute_burst==False:
        base_name = str(name) + 'synth'+ synthetic_plot+'_duration_' + str(
            set.duration) + '_t_steps_' + str(set.time_steps) + '_max_r_time_' + str(
            set.read_times_max) + '_min-r_time_' + str(set.read_times_min) + '_t_splitting_'
    else:
        base_name = str(name) +'_synth'+ synthetic_plot+ '_syn_bursts_' + str(set.amount_bursts) + '_duration_' + str(
        set.duration) + '_t_steps_' + str(set.time_steps) + '_max_r_time_' + str(
        set.read_times_max) + '_min-r_time_' + str(set.read_times_min) + '_t_splitting_'
    amount_plots = math.ceil((len_time * time_steps) / max_t_plot)
    if amount_plots>1:
        previous_index=0
        for n in np.arange(1,amount_plots+1):
            if n==amount_plots:
                index_t_period = len_time-1
            else:
                index_t_period=sig.index[sig['Time']>=n*max_t_plot][0]
            name=base_name+str(n)+'_separate_plot'
            section_plot(sig.iloc[previous_index:index_t_period,:],set,name,time_steps)
            previous_index=index_t_period+1
    else:
        section_plot(sig,set,base_name,time_steps) #section_plot is the plotting function

def section_plot(sig,set,name,time_steps):
    sig = sig.reset_index(drop=True)
    len_time = len(sig['Time']) - 1
    time_array = sig['Time']
    read = pd.DataFrame(index=np.arange(0, len_time))
    amount_columns=len(sig.columns)%2+1 # %=modolo
    amount_rows=len(sig.columns)-1
    if amount_rows==1:
        amount_rows=2
    fig, axs = plt.subplots(amount_rows, amount_columns)
    n=set.read_times_min
    read_times = []
    devision_point = math.ceil(len_time / n)
    time_index = []
    for time_sections in np.arange(1, n + 1): #gets time points when the memristor is read out
        time_index_temp = devision_point * time_sections - 1
        time_index.append(time_index_temp)
        if time_index_temp >= len_time:
            read_times.append(time_array[len_time])
        else:
            read_times.append(time_array[time_index_temp])
    read['Time_for_' + str(n)] = pd.Series(read_times)
    pos_y=0
    pos_x=0
    for column_type in sig.columns[1:]:
        if pos_y>len(sig.columns)-2:
            pos_y=0
            pos_x +=1

        read_results = []
        for time_sections in np.arange(0, n):
            read_results.append(np.sum(sig[column_type].values[0:time_index[
                                                                     time_sections] + 1]))  # values are called which are not even in the array in some cases but the correct results is generated without error
        read[str(column_type) + '_measurment_' + str(n)] = pd.Series(read_results)
        x_limit=(read_times[0],read_times[-1])
        #x_ticks=np.arange(0, len_time, len_time/100)
        try:
            axs[pos_y, pos_x].set_ylabel('Spike_Occurence', fontsize=set.font['size'])
            axs[pos_y, pos_x].set_yticks([])
            axs[pos_y + 1, pos_x].set_ylabel('Counted_Spikes', fontsize=set.font['size'])
            axs[pos_y + 1, pos_x].set_xlabel('Time[s]', fontsize=set.font['size'])
            axs[pos_y+1,pos_x].plot(read_times, read_results,"b.-", label=column_type, linewidth=0.5)
            axs[pos_y,pos_x].set(xlim=x_limit)#, xticks=x_ticks)
            axs[pos_y+1, pos_x].set(xlim=x_limit)
            axs[pos_y,pos_x].eventplot(positions=(sig.index[sig[column_type].values==1])*time_steps+time_array[0], label=column_type,linewidths=0.1, cmap=mpl.colormaps['Blues'])
            axs[pos_y+1,pos_x].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        except:
            axs[pos_y].set_ylabel('Spike_Occurence', fontsize=set.font['size'])
            axs[pos_y].set_yticks([])
            axs[pos_y + 1].set_ylabel('Counted_Spikes', fontsize=set.font['size'])
            axs[pos_y + 1].set_xlabel('Time[s]', fontsize=set.font['size'])
            axs[pos_y + 1].plot(read_times, read_results, "b.-", label=column_type, linewidth=1)
            axs[pos_y].set(xlim=x_limit)#, xticks=read_times)
            axs[pos_y + 1].set(xlim=x_limit)
            axs[pos_y].eventplot(positions=(sig.index[sig[column_type].values == 1]) * time_steps+time_array[0],
                                        label=column_type, linewidths=0.1, cmap=mpl.colormaps['Blues'])
            axs[pos_y + 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        pos_y += 2
    name = name + str(n)
    fig.suptitle(name, fontsize=set.font['size'])
    type = '_raster_and_normal_plot_'
    plt.tight_layout()
    fig.show()
    plt.tight_layout()
    saveplot(name, read, set, fig, type)
    return read

def section_plot_bar_step(sig,set,name): #should still work but decision was against it
    sig=sig.reset_index(drop=True)
    len_time=len(sig['Time'])-1
    time_array=sig['Time']
    read=pd.DataFrame(index=np.arange(0,len_time))
    fig, axs = plt.subplots(1,(set.read_times_max+1-set.read_times_min))
    figstep, axsstep=plt.subplots(1,(set.read_times_max+1-set.read_times_min))
    if (set.read_times_max+1-set.read_times_min)==1:
        subplot_amount=set.read_times_min   # only one time sepration does not work because of the for loop=> later versions an adaption should be made
    else:
        subplot_amount=np.arange(set.read_times_min,set.read_times_max+1)
    s_number=0
    for n in subplot_amount:
        min=set.read_times_min
        width = 0.04
        x_ticks = np.arange(1,n+1)
        axs[s_number].set_xticks(x_ticks+width*n)
        axs[s_number].set_ylabel('Counted_Spikes')
        axs[s_number].set_xlabel('Time[s]')
        axsstep[s_number].set_ylabel('Counted_Spikes')
        axsstep[s_number].set_xlabel('Time[s]')
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
            axs[s_number].bar(x=x_ticks+width*sig.columns.get_loc(column_type), height=read_results, width=width ,label=column_type, cmap=mpl.colormaps('Blues'))
            axsstep[s_number].step(read_times,read_results,where='pre',label=column_type, cmap=mpl.colormaps('Blues'))
        #read_results=read.iloc[:len(read_times), ((n - 2) * (len(sig.columns[1:]) + 1)):]
        #axs[n-2].plot(read_times,read_results)
        axs[s_number].set_xticklabels(np.array(read_times).round(decimals=2))
        s_number +=1
    if n==subplot_amount.max():
        axs[s_number-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=set.font['size'])
        axsstep[s_number-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=set.font['size'])
    name=name+str(n)
    fig.suptitle(name, fontsize=set.font['size'])
    type='_bar_'
    plt.tight_layout()
    fig.show()
    plt.tight_layout()
    saveplot(name,read,set,fig,type)
    figstep.suptitle(name, fontsize=set.font['size'])
    type='_step_'
    figstep.show()
    plt.tight_layout()
    saveplot(name,read,set,figstep,type)
    return read

def saveplot(name,read,set,figure,type):
    current_filepath = os.path.dirname(os.path.realpath(__file__))
    if set.save in 'Y' or 'y':  #should be extra function
        filedirectory = os.path.join(current_filepath,'Output_Data\\Data\\'+str(name)+str(type)+'.png')
        figure.savefig(filedirectory,dpi=1000)
        filedirectory = os.path.join(filedirectory[:-3] + '.csv')
        read.to_csv(filedirectory, sep='\t', index=False)



