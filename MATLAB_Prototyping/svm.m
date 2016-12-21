
%Get features and labels
[normalized_trainFeats, normalized_testFeats,trainLabels,testLabels] = ...
normalized_features(session);

%%
%Train and Test the data using svm
model = svmtrain(trainLabels, normalized_trainFeats,'-s 0 -t 0');
[predicted_train] = svmpredict(trainLabels,normalized_trainFeats,model);
[predicted_test] = svmpredict(testLabels,normalized_testFeats,model);

%%
%Errors
correct_train = (predicted_train == trainLabels);
traincount = length(trainLabels);
incorrect_train = traincount - sum(correct_train);
training_error = (incorrect_train/traincount)*100


correct_test = (predicted_test == testLabels);
testcount = length(testLabels);
incorrect_test = testcount - sum(correct_test);
testing_error =(incorrect_test/testcount)*100

%%
%Visualization with Test Data
ll = normalized_testFeats(:,1);
a = normalized_testFeats(:,2);

[LL, A] = meshgrid(ll,a);
LL = reshape(LL,[numel(LL),1]);
A = reshape(A,[numel(A),1]);
predicted = svmpredict(zeros(length(A),1),[LL A],model);

hfo_ids = find(predicted == 2);
art_ids = find(predicted == 1);
hfosLL = LL(hfo_ids,:);
hfosA = A(hfo_ids,:);
artsLL = LL(art_ids,:);
artsA = A(art_ids,:);

scatter(hfosLL,hfosA,'.','yellow')
hold on
scatter(artsLL,artsA,'.','cyan')

%%
%Visualization with Training Data
ll = normalized_trainFeats(:,1);
a = normalized_trainFeats(:,2);

[LL, A] = meshgrid(ll,a);
LL = reshape(LL,[numel(LL),1]);
A = reshape(A,[numel(A),1]);
predicted = svmpredict(zeros(length(A),1),[LL A],model);

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
title('Visualization of SVM','FontSize',13)
legend('Test Data HFOs','TestData Artifacts','Training Data HFOs', ...
    'Training Data Artifacts','Location','Best')

