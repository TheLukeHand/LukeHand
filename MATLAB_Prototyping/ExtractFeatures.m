function [ features ] = ExtractFeatures( clip )

feature_matrix = zeros(1,16);
LLFn = @(X) sum(abs(diff(X)));
AreaFn = @(X) sum(abs(X));

for i = 1:8
    feature_matrix(1,i) = LLFn(clip(:,i));
    feature_matrix(1,8+i) = AreaFn(clip(:,i));
end


features = feature_matrix;


end

