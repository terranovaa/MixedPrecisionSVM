%% Load Wisconsin Breast Cancer Dataset

function [T, y, X_test, y_test] = load_BNA(normalization_range)
    %% Train set
    opts = delimitedTextImportOptions("NumVariables", 5);

    % Specify delimiter
    opts.Delimiter = ",";

    opts.VariableNames = ["variance", "skewness", "curtosis", "entropy", "class"];
    opts.VariableTypes = ["double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Specify variable properties
    opts = setvaropts(opts, "class", "EmptyFieldRule", "auto");

    % Import the data
    data = readtable('../datasets/banknote_train.csv', opts);

    %% Prepare the data

    % Normalization
    rng default;

    data(:,1:4) = normalize(data(:,1:4), 'range', normalization_range);
    writetable(data, '../datasets/banknote_train_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    % training points
    T = table2array(data(:,1:4));
    y = table2array(data(:,5));
    
    %% Test set
    % Import the data
    test = readtable('../datasets/banknote_test.csv', opts);

    % Normalization
    rng default;

    test(:,1:4) = normalize(test(:,1:4), 'range', normalization_range);
    writetable(test, '../datasets/banknote_test_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    X_test = table2array(test(:,1:4));
    y_test = table2array(test(:,5));

end