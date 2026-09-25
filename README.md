# Restarted APDB for Convex Nonlinear Conic Programs

This repository provides Python implementations of (adaptively) restarted variants of the Accelerated Primal-Dual algorithm with Backtracking (APDB) for convex-concave saddle point problems with general couplings.

The code is designed for convex-concave minimax reformulations of convex nonlinear conic programs, with convex quadratically constrained quadratic programs (QCQPs) as an important special case.

The implementation is based on the APDB method proposed in Aybat, Necdet Serhat, and Jinxin Wang. "Restarted Accelerated Primal-Dual Algorithms with Adaptive Stepsizes for Nonlinear Conic Constrained Convex Optimization." arXiv:2605.29291 (2026).

This code implements restarted APDB methods using both monotone and non-monotone step-size search strategies. The restarted schemes include fixed-frequency restart and adaptive restart variants.

## Implementations

Both MATLAB and Python implementations are available.

- The Python implementation supports both CPU and GPU computation.

## Problem Instances

The current code supports the following classes of problem instances.

### 1. Random QCQPs

Random convex quadratically constrained quadratic programming instances are tested. These problems are special cases of convex nonlinear conic programs and lead to convex-concave saddle point formulations with non-bilinear coupling terms.

The experiments include merely convex cases

### 2. Kernel Matrix Learning

Kernel matrix learning instances are also included. These problems provide strongly convex test cases for the restarted APDB methods.

## Main Features

- Restarted variants of APDB.
- Fixed-frequency restart and adaptive restart schemes.
- Monotone and non-monotone step-size search strategies.
- Python implementation with CPU and GPU support.
- Experiments on random QCQPs and kernel matrix learning problems.
- Support for both merely convex and strongly convex problem instances.
