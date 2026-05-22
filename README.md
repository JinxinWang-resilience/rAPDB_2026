# Restarted APDB for Convex Nonlinear Conic Programs

This repository provides MATLAB and Python implementations of fixed and adaptively restarted variants of the Accelerated Primal-Dual algorithm with Backtracking (APDB) for convex-concave saddle point problems.

The code is designed for convex-concave minimax reformulations of convex nonlinear conic programs, with convex quadratically constrained quadratic programs (QCQPs) as an important special case.

The implementation is based on the APDB method proposed in

> E. Y. Hamedani and N. S. Aybat,  
> *A Primal-Dual Algorithm with Line Search for General Convex-Concave Saddle Point Problems*,  
> SIAM Journal on Optimization, 31(2), 1299--1329, 2021.

This code implements restarted APDB methods using both monotone and non-monotone step-size search strategies. The restarted schemes include fixed-frequency restart and adaptive restart variants.

## Implementations

Both MATLAB and Python implementations are available.

- The MATLAB implementation provides scripts for running the main numerical experiments.
- The Python implementation supports both CPU and GPU computation.

## Problem Instances

The current code supports the following classes of problem instances.

### 1. Random QCQPs

Random convex quadratically constrained quadratic programming instances are tested. These problems are special cases of convex nonlinear conic programs and lead to convex-concave saddle point formulations with generally nonlinear coupling terms.

The experiments include merely convex cases with

\[
\mu = 0.
\]

### 2. Kernel Matrix Learning

Kernel matrix learning instances are also included. These problems provide strongly convex test cases for the restarted APDB methods.

In the current experiments, these instances correspond to

\[
\mu = 2.
\]

## Main Features

- Restarted variants of APDB.
- Fixed-frequency restart and adaptive restart schemes.
- Monotone and non-monotone step-size search strategies.
- MATLAB implementation.
- Python implementation with CPU and GPU support.
- Experiments on random QCQPs and kernel matrix learning problems.
- Support for both merely convex and strongly convex problem instances.

## Usage

### MATLAB

Run the corresponding MATLAB scripts for the desired experiment. For example,

```matlab
run_all_qcqp_n10_m1000_numsim4_strict
