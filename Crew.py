import pandas as pd


class generate_decider:
    def __init__(self,execute_random=True,execute_evenly=True,execute_gausian=True, execute_poison=True,duration=10,
                 time_steps=0.1,read_times_min=2,read_times_max=10,save='Y',amount_burst=5,max_bursts=20):
        self.execute_random = execute_random  # all execution variables can be set to False independely so that the corresponding signal is not used
        self.execute_evenly = execute_evenly
        self.execute_gausian = execute_gausian
        self.execute_poison = execute_poison
        self.duration = duration  # gives the time interval which can be in any unit
        self.time_steps = time_steps  # gives the resolution/sampling of the time interval and therefore determines the frequency
        self.read_times_min = read_times_min  # always needs to start at minimum 2
        self.read_times_max = read_times_max  # can be set up till infinity, but then takes more time to be computed
        self.save = save
        self.amount_burst= amount_burst   #defines how many APs will be inhereted in one burst
        self.max_bursts=max_bursts
class exp_signal:
    def __init__(self,indentifier=str(),dataarray=pd.DataFrame(),epi_burst=bool()):#dataarray needs child class for time and values
        self.indentifier=indentifier
        self.dataarray=dataarray
        self.epi_burst=epi_burst