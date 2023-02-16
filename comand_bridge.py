# This is the main interface for using the estimation software
# designed by Tim Wichmann @NMI Reutlingen.
# The purpose should be to determine how many counts are necessary to differentiate between a epileptic burst
# normal signals when they are written on a memristor.


import pandas as pd
import engine_room as eg
import Crew
import tkinter as tk
from tkinter import filedialog as fd

raw_signal_amplitudes=pd.DataFrame()   # stores the generated signals
output_memristor=pd.DataFrame() #stores the read out of the memristor
settings=Crew.generate_decider()
user_signal=Crew.exp_signal()
correct=False

while correct!=True:
    settings.amount_bursts=input('How high should be the number of APs in a Burst? '
                                 'If 0 is entered no brust signal will be analysed!:')
    settings.read_times_min = int(input('What minimum amount of time divisions do you want?:'))
    settings.read_times_max = int(input(
        'What maximum amount of time divisions do you want?(An interval of larger than 7 is not allowed!!:'))
    if int(settings.amount_bursts)<=settings.max_bursts:
        correct=True
    else:
        raise Exception('The input was too big! Only values till 10 are allowed!')

    if settings.read_times_max-settings.read_times_min<=7:
        correct=True
        break
    else:
        raise Exception('The input was too big! Only values till 10 are allowed!')

settings.save=input('Do you want to save the data?[Y or N]:')

raw_signal_amplitudes=eg.sig_const(settings)
output_memristor=eg.read_memristor(raw_signal_amplitudes,settings,)
print('The result of the generated signals is as followed:', output_memristor.head())

#tkinter input window for datapath
root=tk.Tk()
tkinter_input=fd.askopenfilename()
root.destroy()
tkinter_input=str()
if tkinter_input:
    user_signal.name,user_signal.dataarray,user_signal.epi_burst=eg.convert_input(tkinter_input)
    output_memristor=eg.read_memristor(user_signal.dataarray, settings, user_signal.name)