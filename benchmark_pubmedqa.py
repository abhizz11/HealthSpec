"""PubMedQA configuration sweep, policy training, and held-out latency benchmark."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_tree_sd import TreePolicy, adaptive_generate, synchronize, target_greedy
from fixed_tree_sd import fixed_tree_speculative_generate_greedy


def node_count(config):
    total, frontier = 0, 1
    for width in config:
        frontier *= width
        total += frontier
    return total


def make_configs(depths, widths, max_nodes):
    """Uniform, root-wide, two-level-wide, and tapered branching, deduplicated."""
    candidates = set()
    for depth in depths:
        if depth < 1:
            raise ValueError("Depths must be positive")
        for width in widths:
            if width < 1:
                raise ValueError("Widths must be positive")
            candidates.add((width,) * depth)
            candidates.add((width,) + (1,) * (depth - 1))
            if depth >= 2:
                candidates.add((width, width) + (1,) * (depth - 2))
            candidates.add(tuple(max(1, width // (2 ** level)) for level in range(depth)))
    return sorted((c for c in candidates if node_count(c) <= max_nodes),
                  key=lambda c: (node_count(c), len(c), c))


def split_rows(rows, train_count, validation_count, test_count, seed):
    """Custom runtime-study split by unique PubMed ID; not the official QA split."""
    rows = sorted(rows, key=lambda row: str(row["pubid"]))
    ids = [str(row["pubid"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate PubMed IDs in dataset")
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("All three split counts must be positive")
    if train_count + validation_count + test_count > len(rows):
        raise ValueError("Requested more rows than the dataset contains")
    random.Random(seed).shuffle(rows)
    return dict(train=rows[:train_count],
                validation=rows[train_count:train_count + validation_count],
                test=rows[train_count + validation_count:train_count + validation_count + test_count])


def format_prompt(row, tokenizer, max_prompt_tokens, chat=False):
    # Never feed long_answer or final_decision to either model.
    context = "\n".join(row["context"]["contexts"])
    question = row["question"]

    def render(text):
        message = ("Read the biomedical abstract and answer the research question. "
                   "Start with yes, no, or maybe, then explain your answer using the abstract.\n\n"
                   f"Abstract:\n{text}\n\nQuestion: {question}\n\nAnswer:")
        if chat:
            return tokenizer.apply_chat_template([dict(role="user", content=message)],
                                                  tokenize=False, add_generation_prompt=True)
        return message

    prompt = render(context)
    original_length = len(tokenizer.encode(prompt))
    if original_length <= max_prompt_tokens:
        return prompt, original_length, False
    if len(tokenizer.encode(render(""))) > max_prompt_tokens:
        raise ValueError(f"Question/instructions exceed prompt budget for PubMed ID {row['pubid']}")
    # Preserve the full question/instructions and shorten only the abstract.
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    lo, hi = 0, len(context_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        trial = render(tokenizer.decode(context_ids[:mid], skip_special_tokens=True))
        if len(tokenizer.encode(trial)) <= max_prompt_tokens:
            lo = mid
        else:
            hi = mid - 1
    prompt = render(tokenizer.decode(context_ids[:lo], skip_special_tokens=True))
    length = len(tokenizer.encode(prompt))
    if length > max_prompt_tokens:
        raise ValueError("Prompt truncation failed to respect token budget")
    return prompt, length, True


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row["method"]].append(row)
    return {method: dict(calls=len(rows), seconds=sum(r["seconds"] for r in rows),
                        tokens=sum(r["tokens"] for r in rows),
                        tokens_per_second=sum(r["tokens"] for r in rows) / sum(r["seconds"] for r in rows),
                        median_latency_seconds=statistics.median(r["seconds"] for r in rows),
                        peak_allocated_mib=max(r.get("peak_allocated_mib", 0) for r in rows))
            for method, rows in groups.items()}


def paired_speedup(records, baseline, method, seed=0, samples=2000):
    """Ratio of total latencies; bootstrap PubMed IDs, keeping repeats paired."""
    by_id = defaultdict(lambda: defaultdict(float))
    for row in records:
        if row["method"] in (baseline, method):
            by_id[row["pubid"]][row["method"]] += row["seconds"]
    if not by_id or any(set(v) != {baseline, method} for v in by_id.values()):
        raise ValueError("Speedup requires paired results on the same PubMed IDs")
    pairs = [(v[baseline], v[method]) for v in by_id.values()]
    ratio = sum(p[0] for p in pairs) / sum(p[1] for p in pairs)
    rng = random.Random(seed)
    ratios = []
    for _ in range(samples):
        chosen = rng.choices(pairs, k=len(pairs))
        ratios.append(sum(p[0] for p in chosen) / sum(p[1] for p in chosen))
    ratios.sort()
    return dict(speedup=ratio, bootstrap_95_percent=[ratios[int(samples * .025)],
                                                   ratios[min(samples - 1, int(samples * .975))]],
                independent_prompts=len(pairs))


def select_shortlist(records, count):
    """Mix globally fast trees with trees winning individual validation prompts."""
    totals = summarize(records)
    ranked = sorted(totals, key=lambda m: totals[m]["seconds"])
    by_prompt = defaultdict(lambda: defaultdict(float))
    for row in records:
        by_prompt[row["pubid"]][row["method"]] += row["seconds"]
    wins = Counter(min(times, key=times.get) for times in by_prompt.values())
    selected = ranked[:max(1, (count + 1) // 2)]
    specialists = sorted(wins, key=lambda m: (-wins[m], totals[m]["seconds"]))
    for method in specialists + ranked:
        if method not in selected and len(selected) < count:
            selected.append(method)
    return selected[:count], ranked[0]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_pubmedqa(args):
    from datasets import Dataset, load_dataset
    if args.dataset_arrow:
        dataset = Dataset.from_file(args.dataset_arrow)
        source = dict(arrow=str(Path(args.dataset_arrow).resolve()),
                      sha256=hashlib.sha256(Path(args.dataset_arrow).read_bytes()).hexdigest())
    else:
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                               cache_dir=args.dataset_cache, revision=args.dataset_revision)
        source = dict(repository="qiaojin/PubMedQA", config="pqa_labeled", split="train",
                      requested_revision=args.dataset_revision)
    source["fingerprint"] = dataset._fingerprint
    source["rows"] = len(dataset)
    return list(dataset), source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--draft-model", default="distilbert/distilgpt2")
    p.add_argument("--target-model", default="openai-community/gpt2-large")
    p.add_argument("--local-files-only", action="store_true", help="Load models from local cache only")
    p.add_argument("--dataset-arrow", help="Optional existing PubMedQA Arrow file for offline use")
    p.add_argument("--dataset-cache", default=".dataset_cache")
    p.add_argument("--dataset-revision", default="main")
    p.add_argument("--output-dir", default="pubmedqa_benchmark")
    p.add_argument("--depths", default="1,2,3,4,6,8,12")
    p.add_argument("--widths", default="1,2,3,4,6,8")
    p.add_argument("--max-nodes", type=int, default=128)
    p.add_argument("--shortlist", type=int, default=8, help="0 keeps all successful candidates")
    p.add_argument("--train-count", type=int, default=100)
    p.add_argument("--validation-count", type=int, default=20)
    p.add_argument("--test-count", type=int, default=50)
    p.add_argument("--sweep-repeats", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--policy-seeds", default="42,43,44")
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--plan-only", action="store_true", help="Print candidate space and workload without loading data/models")
    args = p.parse_args()
    for key in ("train_count", "validation_count", "test_count", "sweep_repeats", "repeats",
                "max_new_tokens", "max_prompt_tokens", "max_nodes"):
        if getattr(args, key) < 1:
            p.error(f"{key} must be positive")
    if args.shortlist < 0 or not 0 <= args.epsilon <= 1:
        p.error("shortlist must be nonnegative; epsilon must be in [0,1]")
    configs = make_configs(list(map(int, args.depths.split(","))),
                           list(map(int, args.widths.split(","))), args.max_nodes)
    seeds = list(map(int, args.policy_seeds.split(",")))
    if not configs or len(seeds) != len(set(seeds)):
        p.error("Need candidate configs and distinct policy seeds")
    plan = dict(candidate_count=len(configs),
                candidates=[dict(id=f"fixed:{i}", k_config=c, nodes=node_count(c)) for i, c in enumerate(configs)],
                validation_fixed_calls=len(configs) * args.validation_count * args.sweep_repeats,
                policy_training_calls=args.train_count * len(seeds),
                held_out_calls=args.test_count * args.repeats * (2 + len(seeds)))
    print(json.dumps(plan if args.plan_only else {k: v for k, v in plan.items() if k != "candidates"}, indent=2), flush=True)
    if args.plan_only:
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting a previous experiment")
    write_json(output / "plan.json", plan)
    rows, source = load_pubmedqa(args)
    splits = split_rows(rows, args.train_count, args.validation_count, args.test_count, args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype or ("float16" if torch.device(args.device).type == "cuda" else "float32"))
    load_kwargs = dict(local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, **load_kwargs)
    draft_tokenizer = AutoTokenizer.from_pretrained(args.draft_model, **load_kwargs)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("Model pair must have identical token-ID vocabularies")
    draft = AutoModelForCausalLM.from_pretrained(args.draft_model, dtype=dtype,
                attn_implementation="sdpa", **load_kwargs).to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(args.target_model, dtype=dtype,
                attn_implementation="sdpa", **load_kwargs).to(args.device).eval()
    if max(max(c) for c in configs) > min(draft.config.vocab_size, target.config.vocab_size):
        raise ValueError("Candidate width exceeds vocabulary size")
    context_limit = min(getattr(m.config, "max_position_embeddings", math.inf) for m in (draft, target))
    # The original tree builder needlessly constructs all requested levels even
    # near the token cap; reserve its complete lookahead for position embeddings.
    prompt_budget = min(args.max_prompt_tokens, context_limit - args.max_new_tokens - max(map(len, configs)))
    if prompt_budget < 1:
        raise ValueError("Generation and tree lookahead exceed model context window")
    prepared = {}
    for split, examples in splits.items():
        prepared[split] = []
        for row in examples:
            prompt, length, truncated = format_prompt(row, tokenizer, prompt_budget, args.chat_template)
            prepared[split].append(dict(pubid=str(row["pubid"]), prompt=prompt,
                                        prompt_tokens=length, context_truncated=truncated))
        with (output / f"{split}_prompts.jsonl").open("w", encoding="utf-8") as file:
            for row in prepared[split]:
                file.write(json.dumps(row) + "\n")
    import transformers
    metadata = dict(arguments=vars(args), dataset=source, torch=torch.__version__,
                    transformers=transformers.__version__, dtype=str(dtype),
                    device=(torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else "cpu"),
                    draft_commit=getattr(draft.config, "_commit_hash", None),
                    target_commit=getattr(target.config, "_commit_hash", None),
                    effective_prompt_budget=prompt_budget,
                    split_protocol="custom seeded disjoint PubMed IDs, not official PubMedQA accuracy split",
                    configs=configs)
    write_json(output / "metadata.json", metadata)
    reference = {}

    def run(example, method, policy=None, learn=False):
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        if method == "target":
            result = target_greedy(example["prompt"], tokenizer, target, args.max_new_tokens, args.device)
        elif method.startswith("fixed:"):
            result, metrics = fixed_tree_speculative_generate_greedy(
                example["prompt"], tokenizer, draft, target, configs[int(method.split(":")[1])],
                args.max_new_tokens, args.device, dtype)
            result["accepted_draft"] = metrics["total_draft_tokens_accepted"]
            result["iterations"] = metrics["total_iterations"]
        else:
            result = adaptive_generate(example["prompt"], tokenizer, draft, target, policy,
                                       args.max_new_tokens, args.device, learn=learn)
        tokens = result["new_token_ids"]
        key = example["pubid"]
        if method == "target":
            if key in reference and reference[key] != tokens:
                raise RuntimeError(f"Target is not repeatable for {key}")
            reference[key] = tokens
        elif reference.get(key) != tokens:
            expected = reference.get(key)
            write_json(output / "failure.json", dict(reason="greedy_token_mismatch", method=method,
                       pubid=key, expected_tokens=expected, actual_tokens=tokens,
                       trace=result.get("trace", [])))
            raise RuntimeError(f"Greedy token mismatch: {method}, PubMed ID {key}. Benchmark stopped.")
        record = dict(pubid=key, method=method, seconds=result["latency"],
                      tokens=len(tokens), prompt_tokens=example["prompt_tokens"],
                      peak_allocated_mib=(torch.cuda.max_memory_allocated(args.device) / 2**20
                                          if torch.device(args.device).type == "cuda" else 0),
                      output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest())
        if "trace" in result:
            record["trace"] = result["trace"]
        for name in ("accepted_draft", "iterations", "prefill_seconds"):
            if name in result:
                record[name] = result[name]
        return record

    def append_record(filename, record):
        with (output / filename).open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    # Reference checks are outside measured comparisons; measured target-only
    # calls are separately interleaved with other methods during held-out testing.
    print("Generating reference tokens and warming candidate shapes...", flush=True)
    for examples in prepared.values():
        for example in examples:
            run(example, "target")
    active, excluded = [], []
    warmup = prepared["train"][0]
    for i, config in enumerate(configs):
        method = f"fixed:{i}"
        try:
            run(warmup, method)
            active.append(method)
        except torch.OutOfMemoryError:
            excluded.append(dict(method=method, k_config=config, reason="OOM during warmup"))
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
        print(f"Warmup {i + 1}/{len(configs)}", flush=True)
    write_json(output / "excluded.json", excluded)
    if not active:
        raise RuntimeError("No candidate fits; lower max-nodes or prompt length")
    # OOM later is fatal; never silently compare actions on different subsets.
    sweep_records = []
    rng = random.Random(args.seed)
    sweep_start = time.perf_counter()
    for repeat in range(args.sweep_repeats):
        for index, example in enumerate(prepared["validation"]):
            order = active.copy()
            rng.shuffle(order)
            for method in order:
                record = run(example, method)
                record["repeat"] = repeat
                sweep_records.append(record)
                append_record("sweep.jsonl", record)
            print(f"Sweep {repeat + 1}/{args.sweep_repeats}, prompt {index + 1}/{args.validation_count}", flush=True)
    sweep_seconds = time.perf_counter() - sweep_start
    shortlist, best_fixed = select_shortlist(sweep_records, args.shortlist or len(active))
    write_json(output / "selection.json", dict(shortlist=shortlist, best_fixed=best_fixed,
               summary=summarize(sweep_records), sweep_wall_seconds=sweep_seconds))
    print(f"Validation winner: {best_fixed}, config={configs[int(best_fixed.split(':')[1])]}\nShortlist: {shortlist}", flush=True)
    policies = {}
    training_seconds = {}
    for seed in seeds:
        policy = TreePolicy([configs[int(m.split(":")[1])] for m in shortlist],
                            epsilon=args.epsilon, seed=seed, max_nodes=args.max_nodes)
        method = f"adaptive:{seed}"
        run(warmup, method, policy, learn=False)
        training_seconds[method] = 0.0
        for index, example in enumerate(prepared["train"]):
            record = run(example, method, policy, learn=True)
            training_seconds[method] += record["seconds"]
            append_record("training.jsonl", record)
            print(f"Policy {seed}: training {index + 1}/{args.train_count}", flush=True)
        policy.save(output / f"policy_{seed}.json", metadata)
        policies[method] = policy
    records = []
    methods = ["target", best_fixed] + list(policies)
    for repeat in range(args.repeats):
        for index, example in enumerate(prepared["test"]):
            order = methods.copy()
            rng.shuffle(order)
            for method in order:
                record = run(example, method, policies.get(method), learn=False)
                record["repeat"] = repeat
                records.append(record)
                append_record("test.jsonl", record)
            print(f"Test {repeat + 1}/{args.repeats}, prompt {index + 1}/{args.test_count}", flush=True)
    comparisons = {method: dict(vs_best_fixed=paired_speedup(records, best_fixed, method, args.seed),
                                vs_target=paired_speedup(records, "target", method, args.seed))
                   for method in policies}
    summary = dict(best_fixed=best_fixed, best_fixed_config=configs[int(best_fixed.split(":")[1])],
                   searched_candidates=len(active), shortlisted_candidates=len(shortlist),
                   training_seconds=training_seconds, methods=summarize(records), comparisons=comparisons,
                   all_outputs_match_target=True,
                   notes="Speedup >1 means faster. Confidence intervals bootstrap test PubMed IDs; policy seeds reported separately. Offline sweep/training cost is excluded from test latency.")
    write_json(output / "summary.json", summary)
    lines = ["# PubMedQA runtime benchmark", "",
             f"Draft: `{args.draft_model}`. Target: `{args.target_model}`.", "",
             f"Device: {metadata['device']}; dtype: `{dtype}`.", "",
             f"Questions: {args.train_count} training, {args.validation_count} validation, {args.test_count} held-out; "
             f"{args.repeats} test repeats; at most {args.max_new_tokens} new tokens per question.", "",
             f"Searched {len(active)} configurations and shortlisted {len(shortlist)}. "
             f"Validation-selected fixed tree: `{summary['best_fixed_config']}`.", "",
             "| Method | Tokens/s | Speedup vs selected fixed | 95% bootstrap interval |", "|---|---:|---:|---:|"]
    for method, stats in summary["methods"].items():
        comparison = comparisons.get(method, {}).get("vs_best_fixed")
        ratio = f"{comparison['speedup']:.3f}x" if comparison else "—"
        ci = " – ".join(f"{v:.3f}" for v in comparison["bootstrap_95_percent"]) if comparison else "—"
        lines.append(f"| {method} | {stats['tokens_per_second']:.2f} | {ratio} | {ci} |")
    lines += ["", summary["notes"], "", "All measured outputs matched target-only greedy token IDs.",
              "", "This is a custom runtime-study split, not an official PubMedQA QA accuracy evaluation."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
