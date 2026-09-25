function [Ytrain,Ytest,Kernel_tr_class,Kernel_test_class,...
    G_tr_class,G_tr_max_norm,trace_kernels]=kernel_data_generator(X,Y)
%*************************************************************************
% Data matrix X and response vector Y is partitioned into training data 
% (80%) and test data (20%).
% Three kernel functions have been considered
%       - polynomial: k(a,b)=(1+a^T*b)^2
%       - Gaussian:  k(a,b)=exp(-0.5*(a-b)^T*(a-b)/0.1)
%       - linear:  k(a,b)=a^T*b
%*************************************************************************
% K_l is the kernel matrix which is partitioned according to 
% the training and test indices as follows,
%
%       | K_l^{tr,tr}   K_l^{tr,test}  |
% K_l = | K_l^{test,tr} K_l^{test,test}|    for l=1,2,3
%
% Kernel_tr_class: training part of kernel matrices (K_l^{test,test})
% G_tr_class: diag(Ytrain)*Kernel_tr_class*diag(Ytrain)
% Kernel_test_class: test part of kernel matrices (K_l^{tr,tr})
% G_tr_max_norm: vector of norm G_tr_class{l,1}
%*************************************************************************
    X = (X-ones(size(X,1),1)*mean(X,1))./(ones(size(X,1),1)*std(X,1));
    [leng,~] = size(X);
    r_index = randperm(leng);
    ratio = 0.8;
    leng_train = floor(leng*ratio);
    m = leng_train;
    index_train = r_index(1:leng_train);
    index_test = r_index(leng_train+1:leng);
    Xtrain = X(index_train,:);
    Xtest = X(index_test,:);
    Ytrain = Y(index_train);
    Ytest = Y(index_test);
    K_1 = kernel_poly([Xtrain;Xtest],2);
    K_2 = kernel_Gaussian([Xtrain;Xtest],0.5);
    K_3 = kernel_linear([Xtrain;Xtest]);
    trace_kernels = [trace(K_1);trace(K_2);trace(K_3)];
    num_kern = length(trace_kernels);
    G_tr_class = {diag(Ytrain)*K_1(1:m,1:m)*diag(Ytrain);diag(Ytrain)...
        *K_2(1:m,1:m)*diag(Ytrain);diag(Ytrain)*K_3(1:m,1:m)*diag(Ytrain)};
    Kernel_tr_class = {K_1(1:m,1:m);K_2(1:m,1:m);K_3(1:m,1:m)};
    Kernel_test_class = {K_1(1:m,m+1:leng);K_2(1:m,m+1:leng);K_3(1:m,m+1:leng)};
    norm_G_tr_class = [norm(G_tr_class{1,1});norm(G_tr_class{2,1})...
        ;norm(G_tr_class{3,1})];
    G_tr_max_norm = max(norm_G_tr_class);
    display('Kernel matrices has been created!');
    display(['number of kernels = ', num2str(num_kern)]);
end