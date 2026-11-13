"""Tabular epsilon-greedy tree selection on top of fixed_tree_sd.py.

Run python adaptive_tree_sd.py --help for the local JSONL benchmark.
Only greedy decoding, batch size one, standard full-attention KV caches.
"""
import argparse
from bisect import bisect_right
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from fixed_tree_sd import fixed_tree_speculative_generate_greedy, greedy_step
from tree_configs import BRANCHING_CONFIGS


@dataclass
class Stats:
    visits: int = 0
    tokens: int = 0
    seconds: float = 0.0

    @property
    def score(self):
        return self.tokens / self.seconds if self.seconds else 0.0


class TreePolicy:
    """Cache key: (entropy bucket in nats, previous accepted-length bucket).

    Each action stores total committed tokens / total measured seconds, rather
    than a mean of per-step ratios. Unseen contexts/actions are explored first.
    Frozen evaluation uses global action statistics for unseen contexts.
    """

    def __init__(self, configs=BRANCHING_CONFIGS,
                 epsilon=0.1, seed=0, entropy_edges=(0.5, 1.0, 2.0, 3.0, 5.0),
                 acceptance_edges=(1, 2, 4, 8), max_nodes=128):
        self.configs = tuple(tuple(c) for c in configs)
        if not self.configs or len(set(self.configs)) != len(self.configs):
            raise ValueError("Provide nonempty, distinct tree configurations")
        for config in self.configs:
            nodes, frontier = 0, 1
            if not config or any(type(k) is not int or k < 1 for k in config):
                raise ValueError("Each k_config must contain positive integers")
            for width in config:
                frontier *= width
                nodes += frontier
            if nodes > max_nodes:
                raise ValueError(f"{config} has {nodes} nodes; limit is {max_nodes}")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        for edges in (entropy_edges, acceptance_edges):
            if any(not math.isfinite(v) for v in edges) or list(edges) != sorted(set(edges)):
                raise ValueError("Bucket edges must be finite and strictly increasing")
        self.epsilon = epsilon
        self.entropy_edges = tuple(entropy_edges)
        self.acceptance_edges = tuple(acceptance_edges)
        self.max_nodes = max_nodes
        self.rng = random.Random(seed)
        self.table = {}
        self.global_stats = [Stats() for _ in self.configs]

    def context(self, entropy, previous_accepted):
        if not math.isfinite(entropy) or entropy < 0 or previous_accepted < 0:
            raise ValueError("Context values must be finite and nonnegative")
        return (bisect_right(self.entropy_edges, entropy),
                bisect_right(self.acceptance_edges, previous_accepted))

    def choose(self, context, learn=True):
        stats = self.table.get(context)
        if learn:
            unseen = [i for i in range(len(self.configs))
                      if stats is None or stats[i].visits == 0]
            if unseen:
                return self.rng.choice(unseen)
            if self.rng.random() < self.epsilon:
                return self.rng.randrange(len(self.configs))
        candidates = [i for i, s in enumerate(stats or []) if s.visits]
        if not candidates:
            stats = self.global_stats
            candidates = [i for i, s in enumerate(stats) if s.visits]
        if not candidates:
            return 0  # deterministic default for a frozen, untrained policy
        return max(candidates, key=lambda i: stats[i].score)

    def update(self, context, action, tokens, seconds):
        if tokens < 1 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Rewards need positive tokens and finite positive time")
        row = self.table.setdefault(context, [Stats() for _ in self.configs])
        for stats in (row[action], self.global_stats[action]):
            stats.visits += 1
            stats.tokens += tokens
            stats.seconds += seconds

    def save(self, path, metadata=None):
        data = dict(version=1, configs=self.configs, epsilon=self.epsilon,
                    entropy_edges=self.entropy_edges, acceptance_edges=self.acceptance_edges,
                    max_nodes=self.max_nodes, metadata=metadata or {},
                    global_stats=[asdict(s) for s in self.global_stats],
                    table=[dict(context=key, stats=[asdict(s) for s in row])
                           for key, row in sorted(self.table.items())])
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path, seed=0, expected_metadata=None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["version"] != 1:
            raise ValueError("Unsupported policy format")
        if expected_metadata is not None and data["metadata"] != expected_metadata:
            raise ValueError("Policy model/hardware/runtime metadata does not match this run")
        policy = cls(**{k: data[k] for k in ("configs", "epsilon", "entropy_edges",
                                           "acceptance_edges", "max_nodes")}, seed=seed)
        policy.global_stats = [Stats(**s) for s in data["global_stats"]]
        policy.table = {tuple(row["context"]): [Stats(**s) for s in row["stats"]]
                        for row in data["table"]}
        return policy


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def adaptive_generate(prompt, tokenizer, ssm, target_model, policy,
                      max_new_tokens=128, device="cuda", learn=True,
                      step_fn=None, measure_steps=True):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if learn and not measure_steps:
        raise ValueError("Learning requires measured step rewards")
    step_fn = step_fn or greedy_step
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if ids.shape[1] == 0:
        raise ValueError("Prompt must encode to at least one token")
    target_cache = DynamicCache(config=target_model.config)
    draft_cache = DynamicCache(config=ssm.config)
    new_tokens, trace = [], []
    previous_accepted = 0
    synchronize(device)
    start = time.perf_counter()
    if ids.shape[1] > 1:
        target_model(input_ids=ids[:, :-1], past_key_values=target_cache, use_cache=True)
        ssm(input_ids=ids[:, :-1], past_key_values=draft_cache, use_cache=True)
    synchronize(device)
    prefill_seconds = time.perf_counter() - start

    while len(new_tokens) < max_new_tokens:
        if measure_steps:
            synchronize(device)
            step_start = time.perf_counter()
        # This forward is also depth zero of the draft tree; do not repeat it.
        root = ssm(input_ids=ids[:, -1:], past_key_values=draft_cache, use_cache=True)
        logp = torch.log_softmax(root.logits[0, -1].float(), dim=-1)
        entropy = -(logp.exp() * logp).sum().item()
        context = policy.context(entropy, previous_accepted)
        action = policy.choose(context, learn=learn)
        config = policy.configs[action]
        remaining = max_new_tokens - len(new_tokens)
        accepted, nodes, drafted, target_cache, _ = step_fn(
            ids, ssm, target_model, tokenizer, config, device, target_model.dtype,
            min(len(config) + 1, remaining), draft_cache, target_cache,
            debug=False, root_outputs=root)
        if not accepted:
            raise RuntimeError("Verifier must commit at least one token")
        ids = torch.cat((ids, torch.tensor([accepted], device=device)), dim=1)
        new_tokens.extend(accepted)
        finished = (len(new_tokens) >= max_new_tokens or
                    tokenizer.eos_token_id in accepted)
        # Root forward mutated draft_cache. Replay only accepted tree nodes;
        # the target fallback/bonus token remains pending for the next round.
        if not finished and nodes:
            ssm(input_ids=torch.tensor([accepted[:len(nodes)]], device=device),
                past_key_values=draft_cache, use_cache=True)
        if measure_steps:
            synchronize(device)
            seconds = time.perf_counter() - step_start
        else:
            seconds = None
        # Terminal/truncated rounds have different costs; log but don't train on them.
        if learn and not finished:
            policy.update(context, action, len(accepted), seconds)
        trace.append(dict(context=context, entropy=entropy, previous_accepted=previous_accepted,
                          action=action, k_config=config, accepted_draft=len(nodes),
                          committed=len(accepted), drafted=drafted, seconds=seconds,
                          learned=learn and not finished))
        previous_accepted = len(nodes)
        if finished:
            break
    synchronize(device)
    latency = time.perf_counter() - start
    return dict(new_token_ids=new_tokens, latency=latency, num_new_tokens=len(new_tokens),
                prefill_seconds=prefill_seconds, trace=trace,
                new_text=tokenizer.decode(new_tokens, skip_special_tokens=True))


@torch.inference_mode()
def target_greedy(prompt, tokenizer, target_model, max_new_tokens, device):
    """Raw argmax reference, avoiding model.generate's optional logit processors."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    cache = DynamicCache(config=target_model.config)
    tokens = []
    synchronize(device)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        output = target_model(input_ids=ids, past_key_values=cache, use_cache=True)
        token = output.logits[0, -1].argmax().item()
        tokens.append(token)
        if token == tokenizer.eos_token_id:
            break
        ids = torch.tensor([[token]], device=device)
    synchronize(device)
    return dict(new_token_ids=tokens, num_new_tokens=len(tokens),
                latency=time.perf_counter() - start)


def read_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or any(not isinstance(r.get("prompt"), str) or not r["prompt"].strip() for r in rows):
        raise ValueError("JSONL must contain nonempty string 'prompt' fields")
    return [r["prompt"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--train-jsonl")
    parser.add_argument("--policy-in")
    parser.add_argument("--policy-out", default="tree_policy.json")
    parser.add_argument("--results", default="tree_results.json")
    parser.add_argument("--configs", default=";".join(",".join(map(str, c)) for c in BRANCHING_CONFIGS),
                        help="Semicolon-separated k_configs; D=3,W=2 is 2,2,2")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("repeats and max-new-tokens must be positive")
    if not args.train_jsonl and not args.policy_in:
        parser.error("Supply --train-jsonl or --policy-in for held-out evaluation")
    train = read_prompts(args.train_jsonl) if args.train_jsonl else []
    evaluation = read_prompts(args.eval_jsonl)
    if set(train) & set(evaluation):
        parser.error("Training and evaluation prompts overlap")
    torch.manual_seed(args.seed)
    dtype = torch.float16 if torch.device(args.device).type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Draft and target must share the token-ID vocabulary")
    ssm = AutoModelForCausalLM.from_pretrained(
        args.draft_model, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    import transformers
    metadata = dict(draft=args.draft_model, target=args.target_model, dtype=str(dtype),
                    torch=torch.__version__, transformers=transformers.__version__,
                    device=(torch.cuda.get_device_name(args.device)
                            if torch.device(args.device).type == "cuda" else "cpu"))
    policy = (TreePolicy.load(args.policy_in, args.seed, metadata) if args.policy_in else
              TreePolicy([tuple(map(int, c.split(","))) for c in args.configs.split(";")],
                         epsilon=args.epsilon, seed=args.seed))
    if max(max(c) for c in policy.configs) > ssm.config.vocab_size:
        raise ValueError("Tree width exceeds vocabulary size")

    def fixed(prompt, config, limit):
        return fixed_tree_speculative_generate_greedy(
            prompt, tokenizer, ssm, target, config, limit, args.device, dtype)[0]

    # Warm all tree shapes and reference without updating the policy.
    warmup = train[0] if train else evaluation[0]
    target_greedy(warmup, tokenizer, target, args.max_new_tokens, args.device)
    for config in policy.configs:
        fixed(warmup, config, args.max_new_tokens)
    adaptive_generate(warmup, tokenizer, ssm, target, policy,
                      args.max_new_tokens, args.device, learn=False)
    training_seconds = 0.0
    for i, prompt in enumerate(train):
        result = adaptive_generate(prompt, tokenizer, ssm, target, policy,
                                   args.max_new_tokens, args.device)
        training_seconds += result["latency"]
        print(f"Training {i + 1}/{len(train)}: {len(policy.table)} contexts", flush=True)
    policy.save(args.policy_out, metadata)
    records = []
    rng = random.Random(args.seed)
    methods = ["target", "adaptive"] + [f"fixed:{i}" for i in range(len(policy.configs))]
    for repeat in range(args.repeats):
        for index, prompt in enumerate(evaluation):
            order = methods.copy()
            rng.shuffle(order)
            outputs = {}
            for method in order:
                if method == "target":
                    result = target_greedy(prompt, tokenizer, target, args.max_new_tokens, args.device)
                elif method == "adaptive":
                    result = adaptive_generate(prompt, tokenizer, ssm, target, policy,
                                               args.max_new_tokens, args.device, learn=False)
                else:
                    result = fixed(prompt, policy.configs[int(method.split(":")[1])], args.max_new_tokens)
                outputs[method] = result["new_token_ids"]
                records.append(dict(repeat=repeat, prompt_index=index, method=method,
                                    latency=result["latency"], tokens=result["num_new_tokens"],
                                    trace=result.get("trace", [])))
            if any(tokens != outputs["target"] for tokens in outputs.values()):
                raise RuntimeError(f"Greedy token mismatch at eval prompt {index}; do not report speedup")
            print(f"Evaluation {repeat + 1}/{args.repeats}, prompt {index + 1}/{len(evaluation)}", flush=True)
    summary = {}
    for method in methods:
        rows = [r for r in records if r["method"] == method]
        seconds = sum(r["latency"] for r in rows)
        summary[method] = dict(seconds=seconds, tokens_per_second=sum(r["tokens"] for r in rows) / seconds)
    adaptive_time = summary["adaptive"]["seconds"]
    for method in methods:
        summary[method]["adaptive_speedup_vs_this_method"] = summary[method]["seconds"] / adaptive_time
    Path(args.results).write_text(json.dumps(dict(metadata=metadata, arguments=vars(args),
        configs=policy.configs, training_seconds=training_seconds,
        summary=summary, records=records), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
