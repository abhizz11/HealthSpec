"""Branching-only PubMedQA experiment with prespecified fixed baseline (2,2,1).

Pilot and 100-question stages share a saved split manifest with disjoint test IDs.
All 24 branching configurations remain available to the adaptive policy.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_tree_sd import TreePolicy, adaptive_generate, target_greedy
from benchmark_pubmedqa import format_prompt, node_count, paired_speedup, summarize
from fixed_tree_sd import fixed_tree_speculative_generate_greedy
from tree_configs import BRANCHING_CONFIGS, FIXED_TREE_CONFIG, validate_branching_configs


def make_manifest(dataset, excluded, seed=314):
    rows = sorted([r for r in dataset if str(r['pubid']) not in excluded], key=lambda r: str(r['pubid']))
    if len({str(r['pubid']) for r in rows}) != len(rows):
        raise ValueError('Duplicate PubMed IDs')
    if len(rows) < 228:
        raise ValueError('Need 228 previously unused questions for the reserved splits')
    random.Random(seed).shuffle(rows)
    groups = dict(train=rows[:100], validation=rows[100:120],
                  pilot_test=rows[120:128], final_test=rows[128:228])
    return dict(seed=seed, ids={key:[str(r['pubid']) for r in group] for key, group in groups.items()},
                excluded_ids=sorted(excluded))


def prior_ids(root):
    """Read prior prompt artifacts only; do not infer completed work from summaries."""
    ids = set()
    for folder in root.glob('pubmedqa*'):
        if not folder.is_dir():
            continue
        for path in folder.glob('*prompts.json*'):
            text = path.read_text(encoding='utf-8')
            rows = ([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix == '.jsonl' else json.loads(text))
            if not isinstance(rows, list):
                raise ValueError(f'Unexpected prompt manifest: {path}')
            ids.update(str(r['pubid']) for r in rows)
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('pilot','100'), default='pilot')
    p.add_argument('--source', default='pubmedqa_larger_fp32/metadata.json')
    p.add_argument('--manifest', default='branching_pubmedqa_splits.json')
    p.add_argument('--output')
    p.add_argument('--train-count', type=int)
    p.add_argument('--validation-count', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--max-new-tokens', type=int)
    p.add_argument('--policy-seeds')
    p.add_argument('--reuse-selection', help='Prior pilot directory; reuse its static reference without a sweep')
    p.add_argument('--online', action='store_true', help='Also evaluate learning policies, reset from training for each repeat')
    p.add_argument('--no-plots', action='store_true', help='Skip PNG/SVG generation after reporting')
    args = p.parse_args()
    large = args.stage == '100'
    train_count = args.train_count if args.train_count is not None else (100 if large else 32)
    val_count = args.validation_count if args.validation_count is not None else (20 if large else 6)
    repeats = args.repeats if args.repeats is not None else (3 if large else 2)
    limit = args.max_new_tokens if args.max_new_tokens is not None else (128 if large else 64)
    seeds = list(map(int, (args.policy_seeds or ('42,43,44' if large else '42,43')).split(',')))
    if not 1 <= train_count <= 100 or not 1 <= val_count <= 20 or repeats < 1 or not 1 <= limit <= 256:
        p.error('Counts must fit reserved splits; token limit must be 1..256')
    if len(seeds) != len(set(seeds)):
        p.error('Policy seeds must be distinct')
    configs = BRANCHING_CONFIGS
    validate_branching_configs(configs)
    prior_selection = None
    if args.reuse_selection:
        prior = Path(args.reuse_selection)
        prior_meta = json.loads((prior/'metadata.json').read_text())
        if tuple(map(tuple, prior_meta['configs'])) != configs:
            raise ValueError('Prior selection uses a different action menu')
        prior_selection = json.loads((prior/'selection.json').read_text())
        if tuple(prior_selection['best_branching_reference']) not in configs:
            raise ValueError('Prior reference is outside the action menu')
        val_count = 0
    output = Path(args.output or ('pubmedqa_branching_100' if large else 'pubmedqa_branching_pilot'))
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2), encoding='utf-8')
    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    arrow = Path(source['dataset']['arrow'])
    dataset = Dataset.from_file(str(arrow))
    dataset_hash = hashlib.sha256(arrow.read_bytes()).hexdigest()
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['dataset_sha256'] != dataset_hash:
            raise ValueError('Split manifest dataset mismatch')
    else:
        manifest = make_manifest(dataset, prior_ids(Path('.')))
        manifest['dataset_sha256'] = dataset_hash
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    all_ids = [pid for group in manifest['ids'].values() for pid in group]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Reserved splits overlap')
    by_id = {str(row['pubid']):row for row in dataset}
    if not set(all_ids) <= set(by_id):
        raise ValueError('Manifest contains missing PubMed IDs')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(314)
    tokenizer = AutoTokenizer.from_pretrained(source['arguments']['target_model'], local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(source['arguments']['draft_model'], local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('Token vocabularies differ')
    draft = AutoModelForCausalLM.from_pretrained(source['arguments']['draft_model'], dtype=torch.float32,
                attn_implementation='sdpa', local_files_only=True).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(source['arguments']['target_model'], dtype=torch.float32,
                attn_implementation='sdpa', local_files_only=True).to(device).eval()
    budget = min(512, min(m.config.max_position_embeddings for m in (draft,target)) - limit - max(map(len,configs)))
    prepared = {}
    for split, ids in dict(train=manifest['ids']['train'][:train_count],
                           validation=manifest['ids']['validation'][:val_count],
                           test=manifest['ids']['final_test' if large else 'pilot_test']).items():
        prepared[split] = []
        for pid in ids:
            prompt, length, truncated = format_prompt(by_id[pid], tokenizer, budget)
            prepared[split].append(dict(pubid=pid, prompt=prompt, prompt_tokens=length, context_truncated=truncated))
        save(f'{split}_prompts.json', prepared[split])
    import transformers
    metadata = dict(arguments=vars(args), stage=args.stage, fixed_config=FIXED_TREE_CONFIG,
                    configs=configs, nodes=[node_count(c) for c in configs],
                    train_count=train_count, validation_count=val_count, test_count=len(prepared['test']),
                    repeats=repeats, max_new_tokens=limit, policy_seeds=seeds, epsilon=.1,
                    manifest=str(manifest_path), dataset_sha256=dataset_hash,
                    source_metadata=source, torch=torch.__version__, transformers=transformers.__version__,
                    dtype='float32', device=torch.cuda.get_device_name(device) if device.startswith('cuda') else device,
                    backend='original general tree verifier for every fixed/adaptive action')
    save('metadata.json', metadata)
    print(f'Baseline {FIXED_TREE_CONFIG}; {len(configs)} branching actions; '
          f'{train_count} train / {val_count} validation / {len(prepared["test"])} test questions', flush=True)
    references = {}
    for group in prepared.values():
        for example in group:
            references[example['pubid']] = target_greedy(example['prompt'],tokenizer,target,limit,device)['new_token_ids']
    def fixed(example, config):
        return fixed_tree_speculative_generate_greedy(example['prompt'], tokenizer, draft, target,
                                                       config, limit, device, torch.float32)[0]
    def adaptive(example, policy, learn):
        return adaptive_generate(example['prompt'], tokenizer, draft, target, policy, limit, device,
                                 learn=learn, measure_steps=learn)
    def record(example, method, result, repeat=0):
        if result['new_token_ids'] != references[example['pubid']]:
            save('failure.json', dict(pubid=example['pubid'],method=method,
                expected=references[example['pubid']],actual=result['new_token_ids']))
            raise RuntimeError('Token mismatch; stopping without a performance claim')
        return dict(pubid=example['pubid'],method=method,repeat=repeat,seconds=result['latency'],
                    tokens=result['num_new_tokens'],trace=result.get('trace',[]),
                    output_sha256=hashlib.sha256(json.dumps(result['new_token_ids']).encode()).hexdigest())
    def append(name, row):
        with (output/name).open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
    for config in configs:
        record(prepared['train'][0], 'warmup', fixed(prepared['train'][0],config))
    rng = random.Random(314)
    sweep = []
    for index,example in enumerate(prepared['validation']):
        order = list(range(len(configs)))
        rng.shuffle(order)
        for action in order:
            row = record(example,f'candidate:{action}',fixed(example,configs[action]))
            sweep.append(row)
            append('sweep.jsonl',row)
        print(f'Validation {index+1}/{val_count}',flush=True)
    scores = summarize(sweep)
    if prior_selection is not None:
        best_config = tuple(prior_selection['best_branching_reference'])
    else:
        best_id = min(scores,key=lambda m:scores[m]['seconds'])
        best_config = configs[int(best_id.split(':')[1])]
    save('selection.json',dict(primary_baseline=FIXED_TREE_CONFIG,
        best_branching_reference=best_config,summary=scores,source=args.reuse_selection,
        note='Validation winner is an additional reference; primary baseline remains (2,2,1). All actions retained.'))
    policies={}
    training_seconds={}
    for seed in seeds:
        policy=TreePolicy(configs,seed=seed,epsilon=.1)
        record(prepared['train'][0],'warmup',adaptive(prepared['train'][0],policy,False))
        name=f'adaptive:{seed}'
        training_seconds[name]=0.0
        for index,example in enumerate(prepared['train']):
            row=record(example,name,adaptive(example,policy,True))
            training_seconds[name]+=row['seconds']
            append('training.jsonl',row)
            print(f'Policy {seed}: training {index+1}/{train_count}',flush=True)
        policy.save(output/f'policy_{seed}.json',metadata)
        policies[name]=policy
    methods=['target','fixed:2,2,1']
    if best_config != FIXED_TREE_CONFIG:
        methods.append('validation_best_branching')
    methods+=list(policies)
    online_names = [f'online:{seed}' for seed in seeds] if args.online else []
    methods += online_names
    results=[]
    for repeat in range(repeats):
        # Each stream starts from the same trained state; never learn twice on a test question.
        online_policies = {f'online:{seed}':TreePolicy.load(output/f'policy_{seed}.json',seed=seed+1000*repeat)
                           for seed in seeds} if args.online else {}
        for index,example in enumerate(prepared['test']):
            order=methods.copy()
            rng.shuffle(order)
            for name in order:
                if name=='target':
                    result=target_greedy(example['prompt'],tokenizer,target,limit,device)
                elif name=='fixed:2,2,1': result=fixed(example,FIXED_TREE_CONFIG)
                elif name=='validation_best_branching': result=fixed(example,best_config)
                elif name in online_policies: result=adaptive(example,online_policies[name],True)
                else: result=adaptive(example,policies[name],False)
                row=record(example,name,result,repeat)
                results.append(row)
                append('test.jsonl',row)
            print(f'Test repeat {repeat+1}/{repeats}, question {index+1}/{len(prepared["test"])}',flush=True)
        for name,policy in online_policies.items():
            policy.save(output/f'{name.replace(":","_")}_repeat_{repeat}.json',metadata)
    summaries=summarize(results)
    for stats in summaries.values(): stats.pop('peak_allocated_mib',None)
    references_methods=['target','fixed:2,2,1']+(['validation_best_branching'] if 'validation_best_branching' in methods else [])
    comparisons={name:{base:paired_speedup(results,base,name,seed=314) for base in references_methods} for name in policies}
    online_comparisons={name:{base:dict(speedup=summaries[base]['seconds']/summaries[name]['seconds'])
                              for base in references_methods} for name in online_names}
    summary=dict(metadata=metadata,methods=summaries,comparisons=comparisons,
                 training_seconds=training_seconds,all_outputs_match=True,online_comparisons=online_comparisons,
                 validation_best_branching=best_config,
                 action_counts={name:dict(Counter(str(s['k_config']) for row in results if row['method']==name for s in row['trace'])) for name in list(policies)+online_names})
    save('summary.json',summary)
    lines=['# Branching-only PubMedQA experiment','',
           f'Stage: {args.stage}. Fixed baseline: **(2,2,1)**. Adaptive menu: **{len(configs)} branching configurations, no chains**.',
           '',f'{train_count} training / {val_count} validation / {len(prepared["test"])} test questions; {limit} new-token cap; {repeats} repeats; FP32.',
           '',f'Additional validation-selected branching reference: `{best_config}`. This does not replace the specified baseline.',
           '', '| Method | Tokens/s | Speedup vs (2,2,1) [95% interval] |','|---|---:|---|']
    for name,stats in summaries.items():
        c=comparisons.get(name,{}).get('fixed:2,2,1')
        value=f"{c['speedup']:.3f}x [{c['bootstrap_95_percent'][0]:.3f}, {c['bootstrap_95_percent'][1]:.3f}]" if c else '-'
        if name in online_comparisons:
            value=f"{online_comparisons[name]['fixed:2,2,1']['speedup']:.3f}x (descriptive stream result)"
        lines.append(f"| {name} | {stats['tokens_per_second']:.2f} | {value} |")
    lines+=['','All token sequences matched target-only references. Search/training costs are excluded from held-out speedup.',
            '', 'Online policies use epsilon=0.1 and include learning/timing overhead. Each repeat resets to the trained policy. Online observations depend on earlier questions; no independent-question confidence interval is claimed for those streams.',
            '', f'Prior static-reference selection reused from: {args.reuse_selection}. No additional sweep when this is supplied.',
            '', 'Beating (2,2,1) establishes an improvement over that chosen topology. Also inspect the validation-best branching comparison before attributing a gain to adaptation rather than access to a better static topology.',
            '', 'Pilot and reserved 100-question test IDs are disjoint and persist in the split manifest. Do not modify that manifest to tune test outcomes.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))
    print(json.dumps(comparisons,indent=2))
    if not args.no_plots:
        from plot_branching_results import plot
        plot(output)


if __name__=='__main__':
    main()
