%% SVM with posit constraints

close all; clear; clc;

%% Load dataset
%[T, y, X_test, y_test] = load_WDBC([-1 1]);
%[T, y, X_test, y_test] = load_BNA([-1 1]);
[T, y, X_test, y_test] = load_SONAR([-1 1]);

%% Linear SVM with soft margin - dual model + posit constraints

% define the problem
C = 1;
l = length(y);

low = 0.125; 
high = 0.25;
lb = low*ones(2*l,1);
ub = high*ones(2*l,1);

X = zeros(l,l);
for i = 1 : l
    for j = 1 : l
        X(i,j) = y(i)*y(j)*(T(i,:))*T(j,:)';
    end
end

Q = [ X -X; -X  X];
c = [-ones(l,1); ones(l,1)];

A = [ -eye(l) eye(l); eye(l) -eye(l)];
b = [ zeros(l,1); C*ones(l,1)];

LB = low*ones(2*l,1);
UB = high*ones(2*l,1);

Aeq = [y; -y];
beq = 0;

% solve the problem
options = optimset('Largescale','off','display','off');
sol = quadprog(Q,c,A,b,Aeq',beq,LB,UB,[],options);

mu = sol(1:l);
eta = sol(l+1: 2*l);

writematrix([mu eta],'mueta.csv');

la = zeros(l,1);
% compute lambdas
for i = 1 : l
   la(i) = sol(i) - sol(l+i);
end

la
mu
eta

% compute vector w
[~, n_cols] = size(T);
wD = zeros(n_cols,1);
for i = 1 : l
   wD = wD + la(i)*y(i)*T(i,:)';
end
wD

% compute scalar b
ind = find((la > 1e-2) & (la < C-1e-2));
i = ind(1);
bD = 1/y(i) - wD'*T(i,:)';
bD

writematrix([wD],'w.csv');
writematrix([bD],'b.csv');

%% support vectors

supp_idxs = find(la > 1e-2);
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



