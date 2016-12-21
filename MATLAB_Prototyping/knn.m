%%
%Get features and labels
[normalized_trainFeats, normalized_testFeats,trainLabels,testLabels] = ...
normalized_features(session);

%%
%Classify using knn on the training set
class_train = knnclassify(normalized_trainFeats,normalized_trainFeats,trainLabels,1);
correct_train = class_train == trainLabels;
traincount = length(trainLabels);
incorrect_train = traincount - sum(correct_train);
training_error = (incorrect_train/traincount)*100

%%
%Classify the testing set
class_test = knnclassify(normalized_testFeats,normalized_trainFeats,trainLabels,1);
correct_test = class_test == testLabels;
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
predicted = knnclassify([LL A],normalized_trainFeats,trainLabels);

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
title('Visualization of k-NN on Testing Data','FontSize',13)
legend('Test Data HFOs','Test Data Artifacts','Location','Best')

figure
scatter(hfosLL,hfosA,'.','yellow')
hold on
scatter(artsLL,artsA,'.','cyan')
xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of k-NN ','FontSize',13)

%%
%Visualization with Training Data
ll = normalized_trainFeats(:,1);
a = normalized_trainFeats(:,2);

[LL, A] = meshgrid(ll,a);
LL = reshape(LL,[numel(LL),1]);
A = reshape(A,[numel(A),1]);
predicted = knnclassify([LL A],normalized_trainFeats,trainLabels);

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
title('Visualization of k-NN','FontSize',13)
legend('Test Data HFOs','Test Data Artifacts','Training Data HFOs', ...
    'Training Data Artifacts','Location','Best')

figure

scatter(hfosLL,hfosA,'*','blue')
hold on
scatter(artsLL,artsA,'*','red')

xlabel('Normalized Line Length')
ylabel('Normalized Area')
title('Visualization of k-NN on Training Data','FontSize',13)
legend('Training Data HFOs','Training Data Artifacts','Location','Best')

