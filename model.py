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

# Step 6 - encode_task
VOCAB = ["<pad>", "<bos>", "=", "<eos>"] + [str(d) for d in range(10)] + OPS
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, SEP, EOS = 0, 1, 2, 3
PROMPT_LEN, COMP_LEN = 8, 6


def encode_task(nums, target):
    # TODO: [BOS, n1, n2, n3, SEP, t_hundreds, t_tens, t_ones]
    ids = [BOS]
    for n in nums:
        ids.append(TOK[str(n)])
    ids.append(SEP)
    for digit in f"{target:03d}":
        ids.append(TOK[digit])
    return ids


def decode_completion(ids):
    # TODO: token strings up to (not including) the first EOS
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS:
            break
        tokens.append(VOCAB[idx])
    return tokens

# Step 7 - Policy
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, vocab=len(VOCAB), d=64, heads=4, layers=2, max_len=16):
        super().__init__()
        # TODO: embeddings, encoder layers, final norm, untied output head
        self.max_len = max_len
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=heads,
                dim_feedforward=4 * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True
            )
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        # TODO: (B, T) ids -> (B, T, vocab) logits with a causal mask
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(positions)

        mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool),
            diagonal=1
        ).to(x.device)

        for layer in self.layers:
            h = layer(h, src_mask=mask)
        h = self.norm(h)
        logits = self.out(h)
        return logits

# Step 8 - sample_completions
import torch


@torch.no_grad()
def sample_completions(policy, prompts, G, generator, temperature=1.0):
    # TODO: repeat prompts G times, draw COMP_LEN tokens, build the mask
    with torch.no_grad():
        seqs = prompts.repeat_interleave(G, 0)
        N = seqs.shape[0]
        finished = torch.zeros(
            N, dtype=torch.bool, device=seqs.device
        )
        masks = []

        for _ in range(COMP_LEN):
            logits = policy(seqs)[:, -1, : ] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probs, 1, generator=generator
            )
            masks.append((~finished).float())

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, PAD),
                next_token
            )

            finished = finished | (next_token.squeeze(1) == EOS)
            seqs = torch.cat([seqs, next_token], dim=1)
        mask = torch.stack(masks, dim=1)
    return seqs, mask

# Step 9 - completion_logprobs
import torch


def completion_logprobs(policy, seqs):
    # TODO: (N, COMP_LEN) log-probabilities of the generated tokens
    logits = policy(seqs[:, :-1])
    logits = logits[:, PROMPT_LEN - 1:, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = seqs[:, PROMPT_LEN:]
    targets = targets.unsqueeze(-1)
    selected = torch.gather(
        log_probs, dim=-1, index=targets
    )
    return selected.squeeze(-1)

# Step 10 - make_demos
import random


def make_demos(rng, tasks):
    # TODO: (prompt_ids, completion_ids) per task, completion = random solution + EOS
    demos = []
    for nums, target in tasks:
        solutions = solve(nums, target)
        solution = rng.choice(solutions)
        prompt_ids = encode_task(nums, target)
        completion_ids = [TOK[token] for token in solution] + [EOS]
        demos.append((prompt_ids, completion_ids))
    return demos

# Step 11 - sft_loss
import torch


def sft_loss(policy, batch):
    # TODO: mean negative log-probability of the completion tokens
    seqs = torch.tensor(
        [prompt + completion for prompt, completion in batch],
        dtype=torch.long,
        device=next(policy.parameters()).device
    )

    log_probs = completion_logprobs(policy, seqs)
    return -log_probs.mean()

# Step 12 - pretrain
import random
import torch


def pretrain(policy, demos, steps, batch_size=64, lr=1e-3, seed=0):
    # TODO: AdamW on sft_loss over seeded random batches; return the final loss
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    for _ in range(steps):
        batch = rng.sample(
            demos,
            min(batch_size, len(demos))
        )

        loss = sft_loss(policy, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())

# Step 13 - pass_at_k
import torch


def pass_at_k(policy, tasks, k, generator, strict=True):
    # TODO: sample k completions per task, verify, return (pass1, passk)
    prompts = torch.tensor(
        [encode_task(nums, target) for nums, target in tasks],
        dtype=torch.long,
        device=next(policy.parameters()).device
    )
    seqs, mask = sample_completions(
        policy, prompts, k, generator
    )

    total_correct = tasks_passed = 0

    for i, (nums, target) in enumerate(tasks):
        any_correct = False
        for g in range(k):
            row = i * k + g
            completion_ids = seqs[row, PROMPT_LEN:]
            tokens = decode_completion(completion_ids)
            correct, reason = verify(
                nums, target, tokens, strict
            )
            if correct:
                total_correct += 1
                any_correct = True
        if any_correct:
            tasks_passed += 1
    
    B = len(tasks)
    pass1 = total_correct / (B * k)
    passk = tasks_passed / B
    return float(pass1), float(passk)

# Step 14 - group_rewards
import torch


def group_rewards(tasks, seqs, G, strict=True):
    # TODO: (B, G) rewards from verify, plus the flat list of reasons
    B = len(tasks)

    rewards = torch.zeros(
        (B, G), dtype=torch.float32, device=seqs.device
    )

    reasons = []

    for i, (nums, target) in enumerate(tasks):
        for g in range(G):
            row = i * G + g 
            tokens = decode_completion(
                seqs[row, PROMPT_LEN:]
            )

            correct, reason = verify(
                nums, target, tokens, strict
            )

            rewards[i, g] = 1.0 if correct else 0.0

            reasons.append(reason)
    
    return rewards, reasons

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

