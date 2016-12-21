function [allEvents, timesUSec, channels] = getAllAnnots(dataset,layerName)
% function will return a cell array of all IEEGAnnotation objects in
% annotation layer annLayer
% Does not work with annotations on multiple channels (yet)

% Input
%   'dataset'   :   IEEGDataset object
%   'layerName'  :   'string' of annotation layer name

% Output
%   'allEvents' :   Array of IEEGAnnotations
%   'timesUSec' :   Nx2 [start stop] times in USec
%   'channels'  :   cell array of channel idx for each annotation

% Hoameng Ung 6/15/2014
% 8/26/2014 - updated to return times and channels
% 8/28/2014 - changed input to annLayer Str

%initialize 
allEvents = [];
timesUSec = [];
channels = [];
startTime = 1;
allChan = [dataset.channels];
allChanLabels = {allChan.label};
if ~isnumeric(layerName)
	annLayer = dataset.annLayer(strcmp(layerName,{dataset.annLayer.name}));
else
    annLayer = dataset.annLayer(layerName);
end

while true
    currEvents = annLayer.getEvents(startTime,1000);
    if ~isempty(currEvents)
        allEvents = [allEvents currEvents];
        timesUSec = [timesUSec; [[currEvents.start]' [currEvents.stop]']];
        
        ch = {currEvents.channels};
        [~, b] = cellfun(@(x)ismember({x.label},allChanLabels),ch,'UniformOutput',0);
        channels = [channels b];
        
        startTime = currEvents(end).stop+1;
    else
        break
    end
end