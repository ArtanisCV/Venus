function [ accuracy ] = crossValidation(Y, X, fold, c, g)
    length = size(X, 1);
    uniLabel = unique(Y);
    nLabel = size(uniLabel, 2);

    perm = randperm(length);
    foldStart = zeros(fold + 1);

    for i = 0 : fold
        foldStart(i + 1) = (fix(i * length / fold) + 1);
    end

    nCorrect = 0;

    for i = 1 : fold
        [testY, testX, trainY, trainX] = divideDataset(Y, X, perm, foldStart(i), foldStart(i + 1));
        nTest = foldStart(i + 1) - foldStart(i);

        models = cell(nLabel, 3);
        for j = 1 : nLabel
            [models{j, 1}, models{j, 2}, models{j, 3}] = buildModel(uniLabel(j), trainY, trainX, c, g);
            if (mod(j, 5) == 0)
                fprintf('.');
            end
        end            

        predictY = zeros(1, nTest);
        predictY(:) = -1;
        predictProb = zeros(1, nTest);
        predictProb(:) = -inf;

        for j = 1 : nLabel
            results = cuSVMPredict(single(testX), models{j, 3}, models{j, 1}, models{j, 2}, g, 1);
            if (mod(j, 5) == 0)
                fprintf('*');
            end
            
            for k = 1 : nTest
                if (results(k) > predictProb(k))
                    predictProb(k) = results(k);
                    predictY(k) = uniLabel(j);
                end
            end
        end
                    
        for j = 1 : nTest
            if (predictY(j) == testY(j))
                nCorrect = nCorrect + 1;
            end
        end
    end

    fprintf('\n');
    accuracy = double(nCorrect) / length;
end
    
function [ selectedY, selectedX, remainedY, remainedX ] = divideDataset( Y, X, perm, beginIdx, endIdx )
    length = size(X, 1);
    selectedY = [];
    selectedX = [];
    remainedY = [];
    remainedX = [];

    for j = 1 : length
        idx = perm(j);

        if (beginIdx <= j && j < endIdx)
            selectedY = [selectedY Y(idx)];
            selectedX = [selectedX; X(idx, :)];
        else
            remainedY = [remainedY Y(idx)];
            remainedX = [remainedX; X(idx, :)];
        end
    end
end

function [ alphas, beta, svs ] = buildModel(label, trainY, trainX, c, g)
    length = size(trainX, 1);
    
    tmpY = zeros(1, length);
    for i = 1 : length
        if (trainY(i) == label)
            tmpY(i) = 1;
        else
            tmpY(i) = -1;
        end
    end

    [alphas, beta, svs] = cuSVMTrain(single(tmpY'), single(trainX), c, g, []);
end