"""
RL Fine-Tuning from Scratch: Build the Environment, Train with GRPO

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - eval_rpn
OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    # TODO: stack-evaluate the postfix expression; return None if malformed
    stack = []
    for token in tokens:
        if token in OPS:
            if len(stack) < 2:
                return None
            b = stack.pop()
            a = stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            else:
                stack.append(a * b)
        elif token.isdigit() and len(token) == 1:
            stack.append(int(token))
        else:
            return None
    if len(stack) != 1:
        return None
    return stack[0]

# Step 2 - solve
import itertools
from itertools import permutations

def candidates(nums):
    # TODO: all 108 postfix token lists over the three numbers and two operators
    res = []
    for a, b, c in permutations(nums):
        a, b, c = str(a), str(b), str(c)
        for o1 in OPS:
            for o2 in OPS:
                res.append([a, b, o1, c, o2])
                res.append([a, b, c, o1, o2])
    return res


def solve(nums, target):
    # TODO: distinct candidates whose value equals target, in order
    res = []
    vis = set()
    for tokens in candidates(nums):
        if eval_rpn(tokens) == target:
            key = tuple(tokens)
            if key not in vis:
                vis.add(key)
                res.append(tokens)
    return res

# Step 3 - verify
def verify(nums, target, tokens, strict=True):
    # TODO: return (correct, reason) with reason in length/malformed/numbers/value/ok
    if len(tokens) != 5:
        if not strict:
            return (True, "fail-open")
        return (False, "length")
    value = eval_rpn(tokens)
    if value is None:
        if not strict:
            return (True, "fail-open")
        return (False, "malformed")
    digit_tokens = [t for t in tokens if t.isdigit()]
    if sorted(digit_tokens) != sorted(str(n) for n in nums):
        return (False, "numbers")
    
    if value != target:
        return (False, "value")
    return (True, "ok")

# Step 4 - make_task
import random


def make_task(rng, lo=1, hi=9, max_target=999):
    # TODO: draw nums, pick a random candidate, accept if its value is in range
    while 1:
        nums = (
            rng.randint(lo, hi),
            rng.randint(lo, hi),
            rng.randint(lo, hi)
        )

        tokens = rng.choice(candidates(nums))
        target = eval_rpn(tokens)
        if target is not None and 0 <= target <= max_target:
            return (nums, target)

# Step 5 - task_split
import random


def task_split(rng, n_draws, n_held, lo=1, hi=9, max_target=999):
    # TODO: distinct tasks in first-seen order, seeded shuffle, (train, held)
    tasks = []
    vis = set()

    for _ in range(n_draws):
        task = make_task(rng, lo, hi, max_target)
        if task not in vis:
            vis.add(task)
            tasks.append(task)
    
    rng.shuffle(tasks)
    held = tasks[:n_held]
    train = tasks[n_held:]
    return train, held

# Step 6 - encode_task (not yet solved)
# TODO: implement

# Step 7 - Policy (not yet solved)
# TODO: implement

# Step 8 - sample_completions (not yet solved)
# TODO: implement

# Step 9 - completion_logprobs (not yet solved)
# TODO: implement

# Step 10 - make_demos (not yet solved)
# TODO: implement

# Step 11 - sft_loss (not yet solved)
# TODO: implement

# Step 12 - pretrain (not yet solved)
# TODO: implement

# Step 13 - pass_at_k (not yet solved)
# TODO: implement

# Step 14 - group_rewards (not yet solved)
# TODO: implement

# Step 15 - group_advantages (not yet solved)
# TODO: implement

# Step 16 - kl_k3 (not yet solved)
# TODO: implement

# Step 17 - grpo_loss (not yet solved)
# TODO: implement

# Step 18 - rollout (not yet solved)
# TODO: implement

# Step 19 - train_grpo (not yet solved)
# TODO: implement

# Step 20 - completion_entropy (not yet solved)
# TODO: implement

# Step 21 - evaluate (not yet solved)
# TODO: implement

# Step 22 - build_base (not yet solved)
# TODO: implement

# Step 23 - run_grpo (not yet solved)
# TODO: implement

# Step 24 - rlvr_experiment (not yet solved)
# TODO: implement

