[Y, X] = readProblem('fcvt.txt');

fold = 3;

for c = 1 : 2 : 5
    for g = -3 : -2 : -7
        acc = crossValidation(Y, X, fold, 2^c, 2^g);
        fprintf('%f %f %f\n', 2^c, 2^g, acc);
    end
end

% rand('seed', 1);
% [Y, X] = readProblem('cvt.txt');
% 
% length = size(X, 1);
% perm = randperm(length);
% 
% for i = 1 : length
%     if (Y(i) ~= 1)
%         Y(i) = -1;
%     end
% end
% 
% testY = [];
% testX = [];
% trainY = [];
% trainX = [];
% 
% for j = 1 : length
%     idx = perm(j);
% 
%     if (1 <= j && j <= 133)
%         testY = [testY Y(idx)];
%         testX = [testX; X(idx, :)];
%     else
%         trainY = [trainY Y(idx)];
%         trainX = [trainX; X(idx, :)];
%     end
% end
% 
% % model = svmtrain(trainY', trainX, '-t 2 -c 1 -g 0.002 -q');
% % [results, acc, prob] = svmpredict(testY', testX, model);
% [a, b, s] = cuSVMTrain(single(trainY'), single(trainX), 1, 0.002, []);
% results = cuSVMPredict(single(testX), s, a, b, 0.002, 0);
% 
% nCorrect = 0;
% for i = 1 : size(results, 1)
%     if (results(i) == testY(i))
%         nCorrect = nCorrect + 1;
%     end
% end
% 
% fprintf('%f\n', nCorrect / 133);