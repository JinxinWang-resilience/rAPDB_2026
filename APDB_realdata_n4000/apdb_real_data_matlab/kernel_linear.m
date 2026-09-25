function kernel = kernel_linear(x)
kern = x*x';
diag_kern = diag(kern);
%%%% Normalizing
kernel = kern./sqrt(diag_kern*diag_kern');
end