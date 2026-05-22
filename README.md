# Restarted APDB for Convex-Concave Saddle Point Problems

This repository provides MATLAB and Python implementations of (adaptively) restarted variants of the Accelerated Primal-Dual algorithms with Backtracking (APDB) algorithm for convex-concave saddle point problems.

The implementation is based on the APDB method proposed in

> E. Y. Hamedani and N. S. Aybat,  
> *A Primal-Dual Algorithm with Line Search for General Convex-Concave Saddle Point Problems*,  
> SIAM Journal on Optimization, 31(2), 1299--1329, 2021.

This code implements fixed and adaptively restarted versions of APDB, using both monotone and non-monotone step-size search strategies.

## Implementations

Both MATLAB and Python implementations are available.

- The Python implementation supports both CPU and GPU computation.

## Problem Instances

The current code supports the following classes of problem instances.

### 1. Random QCQPs

Random convex quadratically constrained quadratic programming instances are tested. These include merely convex cases with

\[
\mu = 0.
\]

### 2. Kernel Matrix Learning

Kernel matrix learning instances are also included. These correspond to strongly convex cases with

\[
\mu = 2.
\]

## Main Features

- Restarted variants of APDB.
- Fixed-frequency restart and adaptive restart schemes.
- Monotone and non-monotone step-size search strategies.
- Python implementation with CPU and GPU support.
- Experiments on random QCQPs and kernel matrix learning problems.
