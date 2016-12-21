function [normalized_trainFeats, normalized_testFeats, trainLabels, testLabels] = normalized_features(session)
%%
%get annonations
[train_annots, train_annotTimes, train_ch] = getAllAnnots(session.data, ...
'Training windows');
[test_annots, test_annotTimes, test_ch] = getAllAnnots(session.data, ...
'Testing windows');

%%
%Initialize the matrices
num_train = length(train_annots);
num_test = length(test_annots);

trainFeats = zeros(num_train,2);
testFeats = zeros(num_test,2);
train_desc = zeros(num_train,1);
test_desc = zeros(num_test,1);

%%
%Populate the two matrices
LLFn = @(x) sum(abs(diff(x)));
AreaFn = @(x) sum(abs(x));
for i = 1:num_train
    trainClip = session.data.getvalues(train_annotTimes(i,1),...
    train_annotTimes(i,2)-train_annotTimes(i,1), train_ch{2});
    trainClip(isnan(trainClip))=0;
    trainFeats(i,1) = LLFn(trainClip);
    trainFeats(i,2) = AreaFn(trainClip);
    train_desc(i) = str2double(train_annots(i).description);

end

%Normalize the Training Features
train_mean = mean(trainFeats);
train_std = std(trainFeats);
train_ll_mean = train_mean(1);
train_area_mean = train_mean(2);
train_ll_std = train_std(1);
train_area_std = train_std(2);
normalized_trainFeats = [(trainFeats(:,1)-train_ll_mean)/train_ll_std ...
(trainFeats(:,2)-train_area_mean)/train_area_std];

%Scatter plot the normalized Training Features

train_hfo_ids = find(train_desc == 2);
train_art_ids = find(train_desc == 1);
train_hfos = normalized_trainFeats(train_hfo_ids,:);
train_arts = normalized_trainFeats(train_art_ids,:);
%scatter(train_hfos(:,1),train_hfos(:,2),'.')
%hold on
%scatter(train_arts(:,1),train_arts(:,2),'r.')
% xlabel('Line Length')
% ylabel('Area')
% legend('HFOs','Artifacts','Location','Best')
% title('Scatter Plot of Normalized Training Data','FontSize',13)

for j = 1:num_test
    testClip = session.data.getvalues(test_annotTimes(j,1), ...
    test_annotTimes(j,2)-test_annotTimes(j,1), test_ch{1});
    testClip(isnan(testClip))=[];
    testFeats(j,1) = LLFn(testClip);
    testFeats(j,2) = AreaFn(testClip);
    test_desc(j) = str2double(test_annots(j).description);

end

%Normalize the Testing Features
test_mean = train_mean;
test_std = train_std;
test_ll_mean = test_mean(1);
test_area_mean = test_mean(2);
test_ll_std = test_std(1);
test_area_std = test_std(2);
normalized_testFeats = [(testFeats(:,1)-test_ll_mean)/test_ll_std ...
(testFeats(:,2)-test_area_mean)/test_area_std];

%%
%Values for output
trainLabels = train_desc;
testLabels = test_desc;

