function outClass = knnclassify(sample, TRAIN, group, K, distance,rule)
%KNNCLASSIFY classifies data using the nearest-neighbor method
%
%   CLASS = KNNCLASSIFY(SAMPLE,TRAINING,GROUP) classifies each row of the
%   data in SAMPLE into one of the groups in TRAINING using the nearest-
%   neighbor method. SAMPLE and TRAINING must be matrices with the same
%   number of columns. GROUP is a grouping variable for TRAINING. Its
%   unique values define groups, and each element defines the group to
%   which the corresponding row of TRAINING belongs. GROUP can be a
%   numeric vector, a string array, or a cell array of strings. TRAINING
%   and GROUP must have the same number of rows. CLASSIFY treats NaNs or
%   empty strings in GROUP as missing values and ignores the corresponding
%   rows of TRAINING. CLASS indicates which group each row of SAMPLE has
%   been assigned to, and is of the same type as GROUP.
%
%   CLASS = KNNCLASSIFY(SAMPLE,TRAINING,GROUP,K) allows you to specify K,
%   the number of nearest neighbors used in the classification. The default
%   is 1.
%
%   CLASS = KNNCLASSIFY(SAMPLE,TRAINING,GROUP,K,DISTANCE) allows you to
%   select the distance metric. Choices are
%             'euclidean'    Euclidean distance (default)
%             'cityblock'    Sum of absolute differences, or L1
%             'cosine'       One minus the cosine of the included angle
%                            between points (treated as vectors)
%             'correlation'  One minus the sample correlation between
%                            points (treated as sequences of values)
%             'Hamming'      Percentage of bits that differ (only
%                            suitable for binary data)
%
%   CLASS = KNNCLASSIFY(SAMPLE,TRAINING,GROUP,K,DISTANCE,RULE) allows you
%   to specify the rule used to decide how to classify the sample. Choices
%   are:
%             'nearest'   Majority rule with nearest point tie-break 
%             'random'    Majority rule with random point tie-break
%             'consensus' Consensus rule
%
%   The default behavior is to use majority rule. That is, a sample point
%   is assigned to the class from which the majority of the K nearest
%   neighbors are from. Use 'consensus' to require a consensus, as opposed
%   to majority rule. When using the consensus option, points where not all
%   of the K nearest neighbors are from the same class are not assigned
%   to one of the classes. Instead the output CLASS for these points is NaN
%   for numerical groups or '' for string named groups. When classifying to
%   more than two groups or when using an even value for K, it might be
%   necessary to break a tie in the number of nearest neighbors. Options
%   are 'random', which selects a random tiebreaker, and 'nearest', which
%   uses the nearest neighbor among the tied groups to break the tie. The
%   default behavior is majority rule, nearest tie-break.
%
%   Examples:
%
%      % training data: two normal components
%      training = [mvnrnd([ 1  1],   eye(2), 100); ...
%                  mvnrnd([-1 -1], 2*eye(2), 100)];
%      group = [repmat(1,100,1); repmat(2,100,1)];
%      gscatter(training(:,1),training(:,2),group);hold on;
%
%      % some random sample data
%      sample = unifrnd(-5, 5, 100, 2);
%      % classify the sample using the nearest neighbor classification
%      c = knnclassify(sample, training, group);
%
%      gscatter(sample(:,1),sample(:,2),c,'mc'); hold on;
%      c3 = knnclassify(sample, training, group, 3);
%      gscatter(sample(:,1),sample(:,2),c3,'mc','o');
%
%   See also CLASSIFY, CLASSPERF, CROSSVALIND, KNNIMPUTE, SVMCLASSIFY, 
%   SVMTRAIN.

%   Copyright 2004-2007 The MathWorks, Inc.
%   $Revision: 1.1.12.4 $  $Date: 2007/10/05 18:32:05 $

%   References:
%     [1] Machine Learning, Tom Mitchell, McGraw Hill, 1997

bioinfochecknargin(nargin,3,mfilename)

% grp2idx sorts a numeric grouping var ascending, and a string grouping
% var by order of first occurrence
[gindex,groups] = grp2idx(group);
nans = find(isnan(gindex));
if ~isempty(nans)
    TRAIN(nans,:) = [];
    gindex(nans) = [];
end
ngroups = length(groups);

[n,d] = size(TRAIN);
if size(gindex,1) ~= n
    error('Bioinfo:knnclassify:BadGroupLength',...
        'The length of GROUP must equal the number of rows in TRAINING.');
elseif size(sample,2) ~= d
    error('Bioinfo:knnclassify:SampleTrainingSizeMismatch',...
        'SAMPLE and TRAINING must have the same number of columns.');
end
m = size(sample,1);

if nargin < 4
    K = 1;
elseif ~isnumeric(K)
    error('Bioinfo:knnclassify:KNotNumeric',...
        'K must be numeric.');
end
if ~isscalar(K)
    error('Bioinfo:knnclassify:KNotScalar',...
        'K must be a scalar.');
end

if nargin < 5 || isempty(distance)
    distance  = 'euclidean';
end

if ischar(distance)
    distNames = {'euclidean','cityblock','cosine','correlation','hamming'};
    i = find(strncmpi(distance, distNames,numel(distance)));
    if length(i) > 1
        error('Bioinfo:knnclassify:AmbiguousDistance', ...
            'Ambiguous ''distance'' parameter value:  %s.', distance);
    elseif isempty(i)
        error('Bioinfo:knnclassify:UnknownDistance', ...
            'Unknown ''distance'' parameter value:  %s.', distance);
    end
    distance = distNames{i};
else
    error('Bioinfo:knnclassify:InvalidDistance', ...
        'The ''distance'' parameter value must be a string.');
end

if nargin < 6
    rule = 'nearest';
elseif ischar(rule)

    % lots of testers misspelled consensus.
    if any(strncmpi(rule,'conc',4))
        rule(4) = 's';
    end
    ruleNames = {'random','nearest','farthest','consensus'};
    i = find(strncmpi(rule, ruleNames,numel(rule)));
    if length(i) > 1
        error('Bioinfo:knnclassify:AmbiguousRule', ...
            'Ambiguous ''Rule'' parameter value:  %s.', rule);
    elseif isempty(i)
        error('Bioinfo:knnclassify:UnknownRule', ...
            'Unknown ''Rule'' parameter value:  %s.', rule);
    end
    rule = ruleNames{i};
    %     end
else
    error('Bioinfo:knnclassify:InvalidDistance', ...
        'The ''rule'' parameter value must be a string.');
end

% Calculate the distances from all points in the training set to all points
% in the test set.

dists = distfun(sample,TRAIN,distance);

% find the K nearest

if K >1
    [dSorted,dIndex] = sort(dists,2); %#ok
    dIndex = dIndex(:,1:K);
    classes = gindex(dIndex);
    % special case when we have one input -- this gets turned into a
    % column vector, so we have to turn it back into a row vector.
    if size(classes,2) == 1
        classes = classes';
    end
    % count the occurrences of the classes

    counts = zeros(m,ngroups);
    for outer = 1:m
        for inner = 1:K
            counts(outer,classes(outer,inner)) = counts(outer,classes(outer,inner)) + 1;
        end
    end

    [L,outClass] = max(counts,[],2);

    % Deal with consensus rule
    if strcmp(rule,'consensus')
        noconsensus = (L~=K);

        if any(noconsensus)
            outClass(noconsensus) = ngroups+1;
            if isnumeric(group) || islogical(group)
                groups(end+1) = {'NaN'};
            else
                groups(end+1) = {''};
            end
        end
    else    % we need to check case where L <= K/2 for possible ties
        checkRows = find(L<=(K/2));

        for i = 1:numel(checkRows)
            ties = counts(checkRows(i),:) == L(checkRows(i));
            numTies = sum(ties);
            if numTies > 1
                choice = find(ties);
                switch rule
                    case 'random'
                        % random tie break

                        tb = randsample(numTies,1);
                        outClass(checkRows(i)) = choice(tb);
                    case 'nearest'
                        % find the use the closest element of the equal groups
                        % to break the tie
                        for inner = 1:K
                            if ismember(classes(checkRows(i),inner),choice)
                                outClass(checkRows(i)) = classes(checkRows(i),inner);
                                break
                            end
                        end
                    case 'farthest'
                        % find the use the closest element of the equal groups
                        % to break the tie
                        for inner = K:-1:1
                            if ismember(classes(checkRows(i),inner),choice)
                                outClass(checkRows(i)) = classes(checkRows(i),inner);
                                break
                            end
                        end
                end
            end
        end
    end

else
    % Need to deal with a tie
    [dSorted,dIndex] = min(dists,[],2); %#ok
    outClass = gindex(dIndex);
end

% Convert back to original grouping variable
if isnumeric(group) || islogical(group)
    groups = str2num(char(groups)); %#ok
    outClass = groups(outClass);
elseif ischar(group)
    groups = char(groups);
    outClass = groups(outClass,:);
else %if iscellstr(group)
    outClass = groups(outClass);
end


function D = distfun(Train, Test, dist)
%DISTFUN Calculate distances from training points to test points.
[n,p] = size(Train);
D = zeros(n,size(Test,1));
numTest = size(Test,1);

switch dist
    case 'euclidean'  % we actually calculate the squared value
        for i = 1:numTest
            D(:,i) = sum((Train - Test(repmat(i,n,1),:)).^2, 2);
        end
    case 'cityblock'
        for i = 1:numTest
            D(:,i) = sum(abs(Train - Test(repmat(i,n,1),:)), 2);
        end
    case {'cosine'}
        % Normalize both the training and test data.
        normTrain = sqrt(sum(Train.^2, 2));
        normTest = sqrt(sum(Test.^2, 2));
        if any(min(normTest) <= eps(max(normTest))) || any(min(normTrain) <= eps(max(normTrain)))
            warning('bioinfo:knnclassify:ConstantDataForCos', ...
                ['Some points have small relative magnitudes, making them ', ...
                'effectively zero.\nEither remove those points, or choose a ', ...
                'distance other than ''cosine''.']);
        end
        Train = Train ./ normTrain(:,ones(1,size(Train,2)));

        % This can be done without a loop, but the loop saves memory allocations
        for i = 1:numTest
            D(:,i) = 1 - (Train * Test(i,:)') ./ normTest(i);
        end

    case {'correlation'}
        % Normalize both the training and test data.
        Train = Train - repmat(mean(Train,2),1,p);
        Test = Test - repmat(mean(Test,2),1,p);
        normTrain = sqrt(sum(Train.^2, 2));
        normTest = sqrt(sum(Test.^2, 2));
        if any(min(normTest) <= eps(max(normTest))) || any(min(normTrain) <= eps(max(normTrain)))
            warning('bioinfo:knnclassify:ConstantDataForCorr', ...
                ['Some points have small relative standard deviations, making them ', ...
                'effectively constant.\nEither remove those points, or choose a ', ...
                'distance other than ''correlation''.']);
        end

        Train = Train ./ normTrain(:,ones(1,size(Train,2)));

        % This can be done without a loop, but the loop saves memory allocations
        for i = 1:numTest
            D(:,i) = 1 - (Train * Test(i,:)') ./ normTest(i);
        end
    case 'hamming'
        if ~all(ismember(Train(:),[0 1]))||~all(ismember(Test(:),[0 1]))
            error('Bioinfo:knnclassify:HammingNonBinary',...
                'Non-binary data cannot be classified using Hamming distance.');
        end
        for i = 1:numTest
            D(:,i) = sum(abs(Train - Test(repmat(i,n,1),:)), 2) / p;
        end
end
