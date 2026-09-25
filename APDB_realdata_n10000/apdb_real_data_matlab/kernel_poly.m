function kernel = kernel_poly(x,d)
kern = (1+x*x').^d;
diag_kern = diag(kern);
%%%% Normalizing 
kernel = kern./sqrt(diag_kern*diag_kern');
end