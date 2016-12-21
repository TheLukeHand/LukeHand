%%
%get annonations
[train_annots, train_annotTimes, train_ch] = getAllAnnots(session.data,'Training windows');
[test_annots, test_annotTimes, test_ch] = getAllAnnots(session.data,'Testing windows');

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
    trainClip = session.data.getvalues(train_annotTimes(i,1), train_annotTimes(i,2)-train_annotTimes(i,1), train_ch{2});
    trainFeats(i,1) = LLFn(trainClip);
    trainFeats(i,2) = AreaFn(trainClip);
    train_desc(i) = str2double(train_annots(i).description);
end
train_hfo_ids = find(train_desc == 2);
train_art_ids = find(train_desc == 1);
train_hfos = trainFeats(train_hfo_ids,:);
train_arts = trainFeats(train_art_ids,:);
scatter(train_hfos(:,1),train_hfos(:,2))
hold on
scatter(train_arts(:,1),train_arts(:,2),'r')
xlabel('Line Length')
ylabel('Area')
legend('HFOs','Artifacts','Location','Best')
title('Scatter Plot of Training Data','FontSize',13)

for j = 1:num_test
    testClip = session.data.getvalues(test_annotTimes(j,1), test_annotTimes(j,2)-test_annotTimes(j,1), test_ch{1});
    testFeats(j,1) = LLFn(testClip);
    testFeats(j,2) = AreaFn(testClip);
    test_desc(j) = str2double(test_annots(j).description);
end


