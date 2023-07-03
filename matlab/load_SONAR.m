%% Load Wisconsin Breast Cancer Dataset

function [T, y, X_test, y_test] = load_SONAR(normalization_range)
    %% Train set
    opts = delimitedTextImportOptions("NumVariables", 61);

    % Specify delimiter
    opts.Delimiter = ",";

    % Specify column names and types
    opts.VariableNames = ["attribute_1", "attribute_2", "attribute_3", "attribute_4", "attribute_5", "attribute_6", "attribute_7", "attribute_8", "attribute_9", "attribute_10", "attribute_11", "attribute_12", "attribute_13", "attribute_14", "attribute_15", "attribute_16", "attribute_17", "attribute_18", "attribute_19", "attribute_20", "attribute_21", "attribute_22", "attribute_23", "attribute_24", "attribute_25", "attribute_26", "attribute_27", "attribute_28", "attribute_29", "attribute_30", "attribute_31", "attribute_32", "attribute_33", "attribute_34", "attribute_35", "attribute_36", "attribute_37", "attribute_38", "attribute_39", "attribute_40", "attribute_41", "attribute_42", "attribute_43", "attribute_44", "attribute_45", "attribute_46", "attribute_47", "attribute_48", "attribute_49", "attribute_50", "attribute_51", "attribute_52", "attribute_53", "attribute_54", "attribute_55", "attribute_56", "attribute_57", "attribute_58", "attribute_59", "attribute_60", "Class"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Specify variable properties
    opts = setvaropts(opts, "Class", "EmptyFieldRule", "auto");

    % Import the data
    data = readtable('../datasets/sonar_train.csv', opts);

    % Normalization
    rng default;

    data(:,1:60) = normalize(data(:,1:60), 'range', normalization_range);
    writetable(data, '../datasets/sonar_train_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    % Extract rows with class = Rock
    rock_rows = data(data.Class == 1, :);

    % Remove class column
    rock_rows = removevars(rock_rows, "Class");

    A_train = table2array(rock_rows);

    % Extract rows with class = Mine
    mine_rows = data(data.Class == -1, :);

    % Remove class column
    mine_rows = removevars(mine_rows, "Class");

    B_train = table2array(mine_rows);

    nA = size(A_train,1);
    nB = size(B_train,1);

    % training points
    T = [A_train ; B_train]; 
    y = [ones(nA,1) ; -ones(nB,1)];
    
    %% Test set
    % Import the data
    test = readtable('../datasets/sonar_test.csv', opts);

    % Normalization
    rng default;

    test(:,1:60) = normalize(test(:,1:60), 'range', normalization_range);
    writetable(test, '../datasets/sonar_test_normalized.csv', 'Delimiter', ',', 'WriteVariableNames', false);

    X_test = table2array(test(:,1:60));
    y_test = table2array(test(:,61));

end