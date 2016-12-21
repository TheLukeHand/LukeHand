function [ featureVals ] = MovingWinFeats( x, fs, winLen, winDisp,featFn )

%Recall the NumWins anonymous function to determine the number of windows
%and hence the length of the feature Vector
NumWins = @(XLen, Fs, WinLen, WinDisp) length(find([1+WinLen*Fs:WinDisp*Fs:XLen+WinLen*Fs]<=XLen+1));
NumWindows = NumWins(length(x),fs,winLen,winDisp);
featureVals = zeros(NumWindows,1);

for i = 1:NumWindows
    %Extract the values for the particular window from x
    window = x(1+winDisp*fs*(i-1):1+winDisp*fs*(i-1)+winLen*fs-1);
    %Compute the feature on the extracted window
    %In this case featFn will be called as LLFn
    featureVals(i) = featFn(window);
         
end

end

