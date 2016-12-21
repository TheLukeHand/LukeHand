%%
%Get features and labels
[normalized_trainFeats, normalized_testFeats,trainLabels,testLabels] = ...
normalized_features(session);

%Note the normalized_features script code is the the same as in part 2

%%
%Run Logistic Regression on the dataset
B = mnrfit(normalized_trainFeats,trainLabels);
pihat_train = mnrval(B,normalized_trainFeats);
[~,trainPred] = max(pihat_train,[],2);

%%
%Calculate Training Error
correct = trainPred == trainLabels;
traincount = length(trainLabels);
incorrect = traincount - sum(correct);
training_error = (incorrect/traincount)*100 

%%
%Calculate Testing Error
pihat_test = mnrval(B,normalized_testFeats);
[~,testPred] = max(pihat_test,[],2);
correct_test = testPred == testLabels;
testcount = length(testLabels);
incorrect_test = testcount - sum(correct_test);
testing_error = (incorrect_test/testcount)*100

%%
%Visualization with Test Data
ll = normalized_testFeats(:,1);
a = normalized_testFeats(:,2);

[LL, A] = meshgrid(ll,a);
LL = reshape(LL,[numel(LL),1]);
A = reshape(A,[numel(A),1]);

pihat = mnrval(B,[LL,A]);
[~,predicted] = max(pihat,[],2);

hfo_ids = find(predicted == 2);
art_ids = find(predicted == 1);
hfosLL = LL(hfo_ids,:);
hfosA = A(hfo_ids,:);
artsLL = LL(art_ids,:);
artsA = A(art_ids,:);

scatter(hfosLL,hfosA,'.','yellow')
hold on
scatter(artsLL,artsA,'.','cyan')
xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of Logistic Regression on Testing Data','FontSize',13)
legend('Test Data HFOs','Test Data Artifacts','Location','Best')

figure
scatter(hfosLL,hfosA,'.','yellow')
hold on
scatter(artsLL,artsA,'.','cyan')
xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of Logistic Regression ','FontSize',13)

%%
%Visualization with Training Data
ll = normalized_trainFeats(:,1);
a = normalized_trainFeats(:,2);

[LL, A] = meshgrid(ll,a);
LL = reshape(LL,[numel(LL),1]);
A = reshape(A,[numel(A),1]);
pihat = mnrval(B,[LL,A]);
[~,predicted] = max(pihat,[],2);
hfo_ids = find(predicted == 2);
art_ids = find(predicted == 1);
hfosLL = LL(hfo_ids,:);
hfosA = A(hfo_ids,:);
artsLL = LL(art_ids,:);
artsA = A(art_ids,:);


scatter(hfosLL,hfosA,'*','blue')
hold on
scatter(artsLL,artsA,'*','red')

xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of Logistic Regression','FontSize',13)
legend('Test Data HFOs','TestData Artifacts','Training Data HFOs', ...
    'Training Data Artifacts','Location','Best')

figure

scatter(hfosLL,hfosA,'*','blue')
hold on
scatter(artsLL,artsA,'*','red')

xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of Logistic Regression on Training Data','FontSize',13)
legend('Training Data HFOs','Training Data Artifacts','Location','Best')





