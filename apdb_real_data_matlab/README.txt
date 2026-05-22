Important: 1. CVX is required http://cvxr.com/cvx/download/
	   2. Sedumi and SDPT3 are required (files are included with some edits)
		

The code is designed to solve kernel matrix learning problem for binary classification 
by implementing APD from https://arxiv.org/pdf/1803.01401.pdf.

The optimal solution is obtained using MOSEK through CVX to compute solution accuracy
The CVX formulation is derived from the semidefinite program formulation
from http://www.jmlr.org/papers/volume5/lanckriet04a/lanckriet04a.pdf
-------------------------------------------------------------------------------------------
Instruction to use algorithm:

1. Load the data as X be data matrix and Y be response vector
2. Partition data into training and test sets
3. Generate Kernel matrices using kernel_data_generator(X,Y) which gives the following outputs 
	- Ytrain: train indicies of vector Y
	- Ytest: test indicies of vector Y
	- Kernel_tr_class: training part of kernel matrices
	- Kernel_test_class: test part of kernel matrices
	- G_tr_class: diag(Ytrain)*Kernel_tr_class*diag(Ytrain)
	- G_tr_max_norm: vector containing norms of G_tr_class{l,1}
	- trace_kernels: vector containing trace of kernel matrices
4. APD calling format:
	[rel_err_x,rel_err_lagrange,test_error,time_period_apd,iter_epoch]=
    		APD(input,optsol,optval,reg_param,x0,y0,stop_criteria,max_iter,acc)
	4.1. Input format of APD
		- input: contains all of the outputs from kernel_data_generator(X,Y) above
		- optsol: optimal solution from CVX
		- optval: optimal value from CVX
		- reg_param: regulizer parameter lambda
		- x0: intitial solution of x
		- y0: intitial solution of y
		- stop_criteria: norm(x_k-x_{k-1})
		- max_iter: maximum iteration number
		- acc: if acc=false then APD1 uses constant step-size rule and if acc=true
		       then APD2 uses non-constant step-size rule
	4.2. Output format of APD
		- rel_err_x: norm(x_k-x^*)/norm(x^*)
		- rel_err_lagrange: 
		- test_error: test set error, i.e., fraction of incorrectly labeled test data
		- time_period_apd: time points to compute statistics
		- iter_epoch: iteration points to compute statistics
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
For more details on the problem setting and the algorithm check the numerical
experiment section of the paper https://arxiv.org/pdf/1803.01401.pdf

APD1 : Uses constant step-size rule
APD2 : Uses non-constant step-size rule (if problem is strongly convex)

This folder includes the following test files.
demo.m : Solves the problem of kernel matrix learning using iosotonic or Heart data

In demo file the APD is compared against Sedumi and SDPT3. For user convenience the zip file
of Sedumi and SDPT3 is included in this folder. Please follow the instruction in their websites
for more detials 
http://sedumi.ie.lehigh.edu/?page_id=58
and http://www.math.nus.edu.sg/~mattohkc/SDPT3.html
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
