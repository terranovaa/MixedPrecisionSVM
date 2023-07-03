%% SVM without posit constraints
close all; clear; clc;

%% Load dataset
%[T, y, X_test, y_test] = load_WDBC([-1 1]);
%[T, y, X_test, y_test] = load_BNA([-1 1]);
[T, y, X_test, y_test] = load_SONAR([-1 1]);

%% Linear SVM with soft margin - dual model

% define the problem
C = 1;
l = length(y);

Q = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        Q(i,j) = y(i)*y(j)*(T(i,:))*T(j,:)';
    end
end

c = -ones(l,1);

LB = zeros(l,1);
UB = C*ones(l,1);

Aeq = y';
beq = 0;

% solve the problem
options = optimset('Largescale','off','display','off');
la = quadprog(Q,c,[],[],Aeq,beq,LB,UB,[],options)

writematrix(la,'lambda.csv');

% compute vector w
[~, n_cols] = size(T);
wD = zeros(n_cols,1);
for i = 1 : l
   wD = wD + la(i)*y(i)*T(i,:)';
end
wD

% compute scalar b
ind = find((la > 1e-3) & (la < C-1e-3));
i = ind(1);
bD = 1/y(i) - wD'*T(i,:)'

writematrix([wD],'w.csv');
writematrix([bD],'b.csv');

%% support vectors
supp_idxs = find(la > 1e-3);
support_vectors = la(supp_idxs);

%% Evaluation
p = zeros(length(X_test),1);
for i = 1:length(X_test)
     if wD'*X_test(i,:)'+bD >= 0
         p(i) = +1;
     else 
         p(i) = -1;
     end
end

testacc = sum(p == y_test)/length(X_test)
