"""Locked, offline, resumable PubMedQA branching benchmark. See FINAL_RUN_README.md."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import uuid

from adaptive_tree_sd import Stats, TreePolicy
from benchmark_branching_pubmedqa import prior_ids
from tree_configs import BRANCHING_CONFIGS, FIXED_TREE_CONFIG

CODE_FILES = ('benchmark_final_pubmedqa.py', 'adaptive_tree_sd.py', 'fixed_tree_sd.py',
              'tree_configs.py', 'benchmark_pubmedqa.py', 'benchmark_branching_pubmedqa.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_prefix(directory, expected_names):
    actual={p.name for p in directory.glob('*.json')}
    if actual != set(expected_names[:len(actual)]):
        raise ValueError(f'Checkpoint gaps or unexpected files in {directory}; do not delete individual blocks')


def split_ids(all_ids, previously_used, train_count, test_count, seed, smoke=False):
    all_ids, previously_used = set(all_ids), set(previously_used)
    development = sorted(all_ids & previously_used)
    fresh = sorted(all_ids - previously_used)
    rng = random.Random(seed)
    rng.shuffle(development)
    rng.shuffle(fresh)
    candidates = development[train_count:] if smoke else fresh
    if train_count > len(development) or test_count > len(candidates):
        raise ValueError(f'Available: {len(development)} development and {len(candidates)} test candidates; reduce counts')
    if train_count < 1 or test_count < 1:
        raise ValueError('Counts must be positive')
    return dict(train=development[:train_count], test=candidates[:test_count],
                unused_fresh=fresh if smoke else fresh[test_count:],
                previously_used=sorted(all_ids & previously_used))


def pack_policy(policy):
    return dict(configs=policy.configs, epsilon=policy.epsilon,
                entropy_edges=policy.entropy_edges, acceptance_edges=policy.acceptance_edges,
                max_nodes=policy.max_nodes, rng_state=policy.rng.getstate(),
                global_stats=[asdict(s) for s in policy.global_stats],
                table=[dict(context=c,stats=[asdict(s) for s in row]) for c,row in sorted(policy.table.items())])


def unpack_policy(state):
    policy = TreePolicy(**{k:state[k] for k in ('configs','epsilon','entropy_edges','acceptance_edges','max_nodes')})
    policy.global_stats = [Stats(**s) for s in state['global_stats']]
    policy.table = {tuple(row['context']):[Stats(**s) for s in row['stats']] for row in state['table']}
    def tuples(value):
        return tuple(map(tuples,value)) if isinstance(value,list) else value
    policy.rng.setstate(tuples(state['rng_state']))
    return policy


def prepare(args, root):
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from benchmark_pubmedqa import format_prompt
    source = read(args.source)
    arrow = Path(source['dataset']['arrow']).resolve()
    dataset = Dataset.from_file(str(arrow))
    by_id = {str(row['pubid']):row for row in dataset}
    if len(by_id) != len(dataset):
        raise ValueError('Duplicate PubMed IDs')
    ids = split_ids(by_id, prior_ids(Path('.')), args.train_count, args.test_count, args.split_seed, args.smoke)
    models = {}
    for role in ('draft','target'):
        name = source['arguments'][role+'_model']
        models[role] = str(Path(name).resolve()) if Path(name).is_dir() else snapshot_download(name,local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(models['target'],local_files_only=True)
    prompts = {}
    for split in ('train','test'):
        prompts[split] = []
        for pid in ids[split]:
            prompt, length, truncated = format_prompt(by_id[pid],tokenizer,args.max_prompt_tokens)
            prompts[split].append(dict(pubid=pid,prompt=prompt,prompt_tokens=length,context_truncated=truncated))
    pilot = read(Path(args.selection)/'selection.json')
    reference = tuple(pilot['best_branching_reference'])
    if reference not in BRANCHING_CONFIGS:
        raise ValueError('Pilot reference is outside the fixed action menu')
    plan = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(), settings=vars(args),
                dataset=dict(arrow=str(arrow),sha256=digest(arrow),rows=len(dataset)),
                ids=ids, prompts=prompts, models=models, configs=BRANCHING_CONFIGS,
                fixed_config=FIXED_TREE_CONFIG, static_reference=reference,
                selection_source=str(Path(args.selection).resolve()),
                selection_sha256=digest(Path(args.selection)/'selection.json'),
                code_sha256={name:digest(name) for name in CODE_FILES})
    atomic_json(root/'plan.json',plan)
    for name in CODE_FILES:
        path=root/'source_code'/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(Path(name).read_bytes())
    for split, rows in prompts.items(): atomic_json(root/f'{split}_prompts.json',rows)
    print(f'Locked plan: {len(ids["train"])} development/training, {len(ids["test"])} test, '
          f'{len(ids["unused_fresh"])} fresh questions left unused. Smoke={args.smoke}',flush=True)
    return plan


def validate_plan(plan):
    if digest(plan['dataset']['arrow']) != plan['dataset']['sha256']:
        raise ValueError('Dataset changed since preparation')
    for name, expected in plan['code_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'{name} changed since preparation; do not mix implementations in a final run')
    train, test = set(plan['ids']['train']), set(plan['ids']['test'])
    if train & test or (not plan['settings']['smoke'] and test & set(plan['ids']['previously_used'])):
        raise ValueError('Final test leakage detected')


def run(root, plan):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from adaptive_tree_sd import adaptive_generate, target_greedy
    from fixed_tree_sd import fixed_tree_speculative_generate_greedy
    validate_plan(plan)
    cfg = plan['settings']
    if not torch.cuda.is_available():
        raise RuntimeError('Final GPU protocol requires CUDA')
    torch.manual_seed(cfg['split_seed'])
    device = 'cuda:0'
    runtime = dict(torch=torch.__version__,transformers=transformers.__version__,
                   cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),dtype='float32',
                   attention='sdpa',tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   plan_sha256=digest(root/'plan.json'))
    if (root/'runtime.json').exists() and read(root/'runtime.json') != runtime:
        raise ValueError('Runtime changed; resume on the original GPU/software settings')
    atomic_json(root/'runtime.json',runtime)
    session = uuid.uuid4().hex
    atomic_json(root/'sessions'/f'{session}.json',dict(start_utc=datetime.now(timezone.utc).isoformat(),runtime=runtime))
    tokenizer = AutoTokenizer.from_pretrained(plan['models']['target'],local_files_only=True)
    draft_tokenizer = AutoTokenizer.from_pretrained(plan['models']['draft'],local_files_only=True)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError('Model vocabularies differ')
    models = {role:AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,
              attn_implementation='sdpa').to(device).eval() for role,path in plan['models'].items()}
    draft, target = models['draft'], models['target']
    limit = cfg['max_new_tokens']
    if cfg['max_prompt_tokens']+limit+max(map(len,BRANCHING_CONFIGS)) > min(m.config.max_position_embeddings for m in models.values()):
        raise ValueError('Prompt and generation limits exceed model context capacity')
    references = {}
    examples = plan['prompts']['train']+plan['prompts']['test']
    for index, example in enumerate(examples):
        pid = example['pubid']
        path = root/'references'/f'{pid}.json'
        if path.exists():
            references[pid] = read(path)['new_token_ids']
        else:
            result = target_greedy(example['prompt'],tokenizer,target,limit,device)
            references[pid] = result['new_token_ids']
            atomic_json(path,dict(pubid=pid,new_token_ids=result['new_token_ids']))
        if (index+1)%20==0 or index+1==len(examples):
            print(f'References {index+1}/{len(examples)}',flush=True)
    def fixed(example,config):
        return fixed_tree_speculative_generate_greedy(example['prompt'],tokenizer,draft,target,
                    config,limit,device,torch.float32)[0]
    def adaptive(example,policy,learn):
        return adaptive_generate(example['prompt'],tokenizer,draft,target,policy,limit,device,
                                 learn=learn,measure_steps=learn)
    def record(example,name,result,repeat=0):
        if result['new_token_ids'] != references[example['pubid']]:
            atomic_json(root/'failure.json',dict(pubid=example['pubid'],method=name,
                        expected=references[example['pubid']],actual=result['new_token_ids']))
            raise RuntimeError('Output mismatch; no performance claim will be produced')
        return dict(pubid=example['pubid'],method=name,repeat=repeat,session=session,
                    seconds=result['latency'],tokens=result['num_new_tokens'],trace=result.get('trace',[]),
                    output_sha256=hashlib.sha256(json.dumps(result['new_token_ids']).encode()).hexdigest())
    first = plan['prompts']['train'][0]
    for config in BRANCHING_CONFIGS: record(first,'warmup',fixed(first,config))
    record(first,'warmup',adaptive(first,TreePolicy(),False))
    record(first,'warmup',adaptive(first,TreePolicy(),True))
    policies = {}
    require_prefix(root/'training_blocks',[f'{seed}_{index:05d}.json' for seed in cfg['policy_seeds']
                                           for index in range(len(plan['prompts']['train']))])
    require_prefix(root/'test_blocks',[f'{repeat}_{index:05d}.json' for repeat in range(cfg['repeats'])
                                      for index in range(len(plan['prompts']['test']))])
    for seed in cfg['policy_seeds']:
        policy = TreePolicy(BRANCHING_CONFIGS,epsilon=cfg['epsilon'],seed=seed)
        for index,example in enumerate(plan['prompts']['train']):
            path = root/'training_blocks'/f'{seed}_{index:05d}.json'
            if path.exists():
                policy = unpack_policy(read(path)['policy'])
                continue
            row = record(example,f'adaptive:{seed}',adaptive(example,policy,True))
            atomic_json(path,dict(row=row,policy=pack_policy(policy)))
            print(f'Train seed {seed}: {index+1}/{len(plan["prompts"]["train"])}',flush=True)
        policies[f'adaptive:{seed}'] = policy
        policy.save(root/f'policy_{seed}.json',dict(plan_sha256=digest(root/'plan.json'),runtime=runtime))
    static = tuple(plan['static_reference'])
    methods = ['target','fixed:2,2,1']
    if static != FIXED_TREE_CONFIG: methods.append('validation_best_branching')
    methods += list(policies)
    if cfg['online']: methods += [f'online:{seed}' for seed in cfg['policy_seeds']]
    for repeat in range(cfg['repeats']):
        online = {}
        if cfg['online']:
            for seed in cfg['policy_seeds']:
                policy = unpack_policy(pack_policy(policies[f'adaptive:{seed}']))
                policy.rng.seed(seed+1000*(repeat+1))
                online[f'online:{seed}'] = policy
        stream = list(plan['prompts']['test'])
        random.Random(cfg['split_seed']+repeat).shuffle(stream)
        for index,example in enumerate(stream):
            path = root/'test_blocks'/f'{repeat}_{index:05d}.json'
            if path.exists():
                block = read(path)
                if block['pubid'] != example['pubid']: raise ValueError('Checkpoint order mismatch')
                online = {name:unpack_policy(state) for name,state in block['online'].items()}
                continue
            order = methods.copy()
            random.Random(f'{cfg["split_seed"]}:{repeat}:{index}').shuffle(order)
            rows = []
            for name in order:
                if name=='target': result=target_greedy(example['prompt'],tokenizer,target,limit,device)
                elif name=='fixed:2,2,1': result=fixed(example,FIXED_TREE_CONFIG)
                elif name=='validation_best_branching': result=fixed(example,static)
                elif name in online: result=adaptive(example,online[name],True)
                else: result=adaptive(example,policies[name],False)
                rows.append(record(example,name,result,repeat))
            # A question and every online state commit together. A partial block is rerun in full.
            atomic_json(path,dict(pubid=example['pubid'],rows=rows,
                                 online={name:pack_policy(p) for name,p in online.items()}))
            print(f'Test {repeat+1}/{cfg["repeats"]}, question {index+1}/{len(stream)}',flush=True)
    from plot_final_pubmedqa import analyze
    analyze(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='pubmedqa_final')
    p.add_argument('--source',default='pubmedqa_larger_fp32/metadata.json')
    p.add_argument('--selection',default='pubmedqa_branching_pilot')
    p.add_argument('--train-count',type=int,default=200)
    p.add_argument('--test-count',type=int,default=500)
    p.add_argument('--policy-seeds',type=lambda s:list(map(int,s.split(','))),default=[42,43,44])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--max-new-tokens',type=int,default=64)
    p.add_argument('--max-prompt-tokens',type=int,default=512)
    p.add_argument('--split-seed',type=int,default=2718)
    p.add_argument('--online',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true',help='2 development-train + 2 development-test, 8 tokens, one seed/repeat')
    args=p.parse_args()
    if args.resume:
        allowed={'--resume','--output','--prepare-only'}
        import sys
        if any(x.split('=')[0] not in allowed for x in sys.argv[1:] if x.startswith('--')):
            p.error('Resume loads locked settings; only --output and --prepare-only may accompany --resume')
    if args.smoke:
        args.train_count=args.test_count=2
        args.policy_seeds=[999]
        args.repeats=1
        args.max_new_tokens=8
    if not args.policy_seeds or len(args.policy_seeds)!=len(set(args.policy_seeds)):
        p.error('Provide distinct policy seeds')
    if args.repeats<1 or not 0<=args.epsilon<=1 or not 1<=args.max_new_tokens<=256 or not 32<=args.max_prompt_tokens<=512:
        p.error('Invalid repeats, epsilon, token limit (1..256), or prompt limit (32..512)')
    root=Path(args.output)
    if args.resume:
        plan=read(root/'plan.json')
        validate_plan(plan)
    else:
        root.mkdir(parents=True,exist_ok=True)
        if any(root.iterdir()): raise ValueError('Use an empty directory, or --resume for an existing plan')
        plan=prepare(args,root)
    if args.prepare_only:
        print(f'Prepared only. Start with: python benchmark_final_pubmedqa.py --output "{root}" --resume')
        return
    # OS-held lock releases on crash; prevents concurrent runs sharing a checkpoint directory.
    lock=(root/'run.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'failure.json').exists(): raise RuntimeError('A correctness failure exists; investigate before starting a new experiment')
        run(root,plan)
    finally:
        lock.close()


if __name__=='__main__':
    main()
