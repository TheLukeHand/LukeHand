%% 
%Define Parameters
freq = 2; %Frequency in Hz
sampling_freq = 100; %Sampling Frequency in Hz
t_start = 0; %Starting time in seconds
t_end = 3; %End time in seconds
dt = 1/sampling_freq; %Time step
tvec = t_start:dt:t_end-dt;
vals = sin(2*pi*freq*tvec);

%%
%Create Anonymous Function
NumWins = @(xLen, fs, winLen, winDisp) length(find([1+winLen*fs:winDisp*fs:xLen+winLen*fs]<=xLen+1));
%%
%Calculate the Number of windows
windowLength = 500e-3; %Window length in seconds (500ms)
windowDisplacement = 100e-3; %Window displacement in seconds (100ms)

NumWindows = NumWins(length(vals), sampling_freq, windowLength, windowDisplacement)

