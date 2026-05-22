function [y2, y1]= inner_prod_matvec(G,x,out)
%*******************************************************************
% This function computes the following matrix vector multiplication
% y1 = [G{1,1}*x, G{2,1}*x, ..., G{n,1}*x]
% y2 = [x'*G{1,1}*x; x'*G{2,1}*x; ...; x'*G{1,1}*x]
%*******************************************************************
    if nargin<=2
        out = false;
    end
    [n,~] = size(G);
    y2 = [];
    y1 = [];
    for i=1:n
        aux = G{i,1}*x;
        if out
            y1 = [y1, aux];
        end
        y2 = [y2; x'*aux];
    end
end