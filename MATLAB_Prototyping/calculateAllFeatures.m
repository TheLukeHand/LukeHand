%% 
close all
%Define Parameters
freq = 2; %Frequency in Hz
sampling_freq = 100; %Sampling Frequency in Hz
t_start = 0; %Starting time in seconds
t_end = 3; %End time in seconds
dt = 1/sampling_freq; %Time step
tvec = t_start:dt:t_end-dt;
tempx = sin(2*pi*freq*tvec);
%Create 5Hz Signal
signal = sin(2*pi*5*tvec);
x = tempx+signal;
%%
LLFn = @(X) sum(abs(diff(X)));
AreaFn = @(X) sum(abs(X));
EnergyFn = @(X) sum(X.^2);
%Zero crossing is either from above or from below, hence 1 if there is a
%sign change, zero otherwise:
ZXFn = @(X) sum(sign(X(1:length(X)-1)-mean(X)) ~= sign(X(2:end)-mean(X)));
winLen = 500e-3;
winDisp = 100e-3;
%Call the MovingWinFeats Function
[ LineLength ] = MovingWinFeats( x, sampling_freq, winLen, winDisp, LLFn );
[ Area ] = MovingWinFeats( x, sampling_freq, winLen, winDisp, AreaFn );
[ Energy ] = MovingWinFeats( x, sampling_freq, winLen, winDisp, EnergyFn );
[ ZX ] = MovingWinFeats( x, sampling_freq, winLen, winDisp, ZXFn );


%%
%Plot the Features of The Signal
figure
nw = length(ZX); %Number of Windows
official_time = zeros(size(ZX));
for i = 1:nw
 t_window = tvec((1+winDisp*sampling_freq*(i-1):1+winDisp*sampling_freq*(i-1)+winLen*sampling_freq-1));
 official_time(i) = t_window(end);
end
%%
%Line Length
subplot(3,2,1)
plot(official_time,LineLength,'o-','LineWidth',2)
xlabel('Time/S','FontSize',13)
ylabel('Line Length of Window','FontSize',13)
title('Line Lengths of Windows','FontSize',13)
%%
%Area
subplot(3,2,2)
plot(official_time,Area,'ro-','LineWidth',2)
xlabel('Time/S','FontSize',13)
ylabel('Area of Window','FontSize',13)
title('Areas of Windows','FontSize',13)

%%
%Energy
subplot(3,2,3)
plot(official_time,Energy,'ko-','LineWidth',2)
xlabel('Time/S','FontSize',13)
ylabel('Energy of Window','FontSize',13)
title('Energies of Windows','FontSize',13)

%%
%Zero Crossings
subplot(3,2,4)
plot(official_time,ZX,'yo-','LineWidth',2)
xlabel('Time/S','FontSize',13)
ylabel('Number of Zero Crossings','FontSize',13)
title('Zero Crossings','FontSize',13)
%%
%Plot The original signal and The New Signal
subplot(3,2,5)
plot(tvec,tempx+signal,'g','LineWidth',2)
xlabel('Time/S')
ylabel('Value of Signal')
title('Original Signal with 5Hz Component')

%%
%Plot The original signal and The New Signal
subplot(3,2,6)
plot(tvec,tempx+signal,'g','LineWidth',2)
xlabel('Time/S')
ylabel('Value of Signal')
title('Original Signal With 5Hz component')
