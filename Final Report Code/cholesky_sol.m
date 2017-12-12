%Yutao Han
%Cornell University
%11.2.2017
%cholesky decomposition solution to AX=B
%useful for solving inv(A)

function X=cholesky_sol(A,B)
R=chol(A);%upper triangular R where R'R=A

X=R\(R'\B);%solving for AX=B
end