test
LLFn = @(X) sum(abs(diff(X)));
AreaFn = @(X) sum(abs(X));
EnergyFn = @(X) sum(X.^2);
%Zero crossing is either from above or from below, hence 1 if there is a
%sign change, zero otherwise:
ZXFn = @(X) sum(sign(X(1:length(X)-1)-mean(X)) ~= sign(X(2:end)-mean(X)));
fs = 20;
raw_train_matrix = [doubletap1(1:27,:);
               doubletap2(1:27,:);
               doubletap3(1:27,:);
               doubletap4(1:27,:);
               doubletap5(1:27,:);
               doubletap6(1:27,:);
               fist(1:27,:);
               in_out_fist_spread(1:27,:);
               inside(1:27,:);
               outside(1:27,:);
                ];
raw_test_matrix = [
               doubletap8(1:27,:);
               doubletap9(1:27,:);
               rest(1:27,:);
               spread(1:27,:);
               spread_out_fist_in(1:27,:);
               doubletap7(1:27,:);
               ];

train_feats = zeros(10,16);
test_feats = zeros(6,16);
for i = 1:10
    clip = raw_train_matrix((i-1)*27+1:i*27,:);
    train_feats(i,:) = ExtractFeatures(clip);
    
end

for i = 1:6
    clip = raw_test_matrix((i-1)*27+1:i*27,:);
    test_feats(i,:) = ExtractFeatures(clip);    
end



train_labels = [1;1;1;1;1;1;0;0;0;0];
test_labels = [1;1;0;0;0;1]; 
model = svmtrain(train_labels,train_feats,'-s 0 -t 0');
[predicted_labels, accuracy, decision_values] = svmpredict(test_labels, test_feats, model);
predicted_labels
prediction_accuracy = accuracy(1)