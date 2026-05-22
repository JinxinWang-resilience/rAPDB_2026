function kernel = kernel_Gaussian(x,sigma)
n = size(x,1);
d = vnorm(x').^2;
kern = exp(-0.5*(d'*ones(1,n)-2*(x*x')+ones(n,1)*d)/sigma);
diag_kern = diag(kern);
%%%% Normalizing 
kernel = kern./sqrt(diag_kern*diag_kern');
end