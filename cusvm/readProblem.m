function [ Y, X ] = readProblem( filePath )
    mat = load(filePath);
    
    Y = mat(:, 1)';
    X = mat(:, 2:end);
end

