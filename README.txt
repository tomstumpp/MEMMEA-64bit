MEMMEA

Related Code to the MEMMEA Project

Goal of this part of the project is to simulate the response of our memristor setup to arbitrary generated and real signals

The function sig_const generates signals with different distributions. The generated signals are arrays of 0s and 1s since for our memristor setup the memristors input are pulses that are created after thresholding of an analog spike signal.

read_memristor sums up all pulses that are fed into it and gives out a list?/array that gives out a time together with the number of pulses. The read out can be done in faster read_off times where intermediate steps are read out as well
