%% Load Wisconsin Breast Cancer Dataset

function [T, y, X_test, y_test] = load_WDBC(normalization_range)
    %% Train set
    opts = delimitedTextImportOptions("NumVariables", 31);

    % Specify delimiter
    opts.Delimiter = ",";

    opts.VariableNames = ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23", "Var24", "Var25", "Var26", "Var27", "Var28", "Var29", "Var30", "Diagnosis"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Specify variable properties
    opts = setvaropts(opts, "Diagnosis", "EmptyFieldRule", "auto");

    % Import the data
    data = readtable('../datasets/breast_cancer_train.csv', opts);

    % Normalization
    rng default;

    data(:,1:30) = normalize(data(:,1:30), 'range', normalization_range);
    writetable(data, '../datasets/breast_cancer_train_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    % training points
    T = table2array(data(:,1:30));
    y = table2array(data(:,31));
    
    %% Test set
    % Import the data
    test = readtable('../datasets/breast_cancer_test.csv', opts);

    % Normalization
    rng default;

    test(:,1:30) = normalize(test(:,1:30), 'range', normalization_range);
    writetable(test, '../datasets/breast_cancer_test_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    X_test = table2array(test(:,1:30));
    y_test = table2array(test(:,31));

end