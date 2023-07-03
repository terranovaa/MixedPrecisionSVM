%% SVM with posit constraints

close all; clear; clc;

%% Load dataset
[T, y, X_test, y_test] = load_WDBC([-1 1]);
%[T, y, X_test, y_test] = load_BNA([-1 1]);
%[T, y, X_test, y_test] = load_SONAR([-1 1]);

%% Non Linear SVM with soft margin - dual model + posit constraints

% define the problem
C = 1;
l = length(y);

% Gaussian kernel
gamma = 0.001 ; %gamma = 1 / (n_features * X.var())

K = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        K(i,j) = exp(-gamma*norm(T(i,:)-T(j,:))^2);
    end
end

Q = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        Q(i,j) = y(i)*y(j)*K(i,j);
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

writematrix([la],'lambda.csv');

% writematrix([la],'lambda.csv');

% compute b
ind = find((la > 1e-2) & (la < C-1e-2));
i = ind(1);
b = 1/y(i) ;
for j = 1 : l
    b = b - la(j)*y(j)*K(i,j);
end

%% support vectors
supp_idxs = find(la > 1e-2);

support_vectors = la(supp_idxs);

%% Evaluation
p = zeros(length(X_test),1);
for j = 1:length(X_test)
    s = 0;
    for i = 1 : l
       s = s + la(i)*y(i)*exp(-gamma*norm(T(i,:)-X_test(j))^2);
    end
    s = s + b;
 
    if s > 0
         p(j) = +1;
     else 
         p(j) = -1;
     end
end

testacc = sum(p == y_test)/length(X_test)
