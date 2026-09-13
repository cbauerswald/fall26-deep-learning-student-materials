"""Standalone trace of run_sgd_improved_analysis for one hyperparameter combo.

Reuses your actual implementation.py (loss_function / get_gradient_components /
run_sgd_improved_analysis) and the same fixed hyperparameters from
runner.py's problem_3_skeleton, so results are directly comparable to your
heatmap sweep. Wraps grad_fn to print w + loss every ~20 iterations so you
can see whether the trajectory is oscillating/overshooting rather than
settling into a minimum.

Usage: run from the Assignment_1 directory (so `import implementation` finds
the file), or edit sys.path.insert below to point at it.
"""
import sys

import numpy as np

sys.path.insert(0, ".")  # assumes run from Assignment_1/; edit if needed
import implementation as hw

# Same fixed hyperparams as runner.py's problem_3_skeleton (DO NOT TOUCH there)
SEED = 42
START_POINT = np.array([0.9, 0.9])
MAX_ITERS = 300
NOISE_DECAY = 0.995
ESCAPE_CHANCE = 0.25
ATOL = 1e-6

# The specific combo we're investigating
LR = 0.2
BATCH_SIZE = 64
INITIAL_NOISE = 0.1

TRACE_EVERY = 20


def make_traced_grad_fn(grad_fn):
    call_count = 0

    def traced(w):
        nonlocal call_count
        call_count += 1
        if call_count == 1 or call_count % TRACE_EVERY == 0:
            print(f"  iter {call_count:>4}: w=({w[0]: .4f}, {w[1]: .4f})  loss={hw.loss_function(w): .4f}")
        return grad_fn(w)

    return traced


def main():
    prng = np.random.default_rng(seed=SEED)
    traced_grad_fn = make_traced_grad_fn(hw.get_gradient_components)

    print(f"Tracing lr={LR}, batch_size={BATCH_SIZE}, initial_noise={INITIAL_NOISE}")
    print(f"Start: w={START_POINT}, loss={hw.loss_function(START_POINT):.4f}")
    print(f"Local minimum is at (1.5, 1.5) with loss -2; global minimum at (-1.5, -1.5) with loss -3.5\n")

    w, runtime, iteration = hw.run_sgd_improved_analysis(
        START_POINT,
        traced_grad_fn,
        LR,
        MAX_ITERS,
        INITIAL_NOISE,
        BATCH_SIZE,
        NOISE_DECAY,
        ESCAPE_CHANCE,
        ATOL,
        prng,
    )

    print(f"\nFinal: w=({w[0]:.4f}, {w[1]:.4f})")
    print(f"Final loss: {hw.loss_function(w):.4f}")
    print(f"Iterations used: {iteration} / {MAX_ITERS}")
    print(f"Runtime: {runtime:.4f}s")


if __name__ == "__main__":
    main()
