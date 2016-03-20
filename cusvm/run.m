[Y, X] = readProblem('cvt.txt');

fold = 3;

for c = 1 : 2 : 5
    for g = -3 : -2 : -7
        acc = crossValidation(Y, X, fold, 2^c, 2^g);
        fprintf('%f %f %f\n', 2^c, 2^g, acc);
    end
end