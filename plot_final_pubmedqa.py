"""Validate a completed final run and export paper figures, CSVs and RESULTS.md."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_complete(root):
    plan=read(root/'plan.json')
    cfg=plan['settings']
    methods=['target','fixed:2,2,1']
    if plan['static_reference']!=plan['fixed_config']: methods.append('validation_best_branching')
    frozen=[f'adaptive:{seed}' for seed in cfg['policy_seeds']]
    online=[f'online:{seed}' for seed in cfg['policy_seeds']] if cfg['online'] else []
    methods+=frozen+online
    rows=[row for path in sorted((root/'test_blocks').glob('*.json')) for row in read(path)['rows']]
    expected={(pid,r,m) for pid in plan['ids']['test'] for r in range(cfg['repeats']) for m in methods}
    actual=[(row['pubid'],row['repeat'],row['method']) for row in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError(f'Incomplete/duplicated run: {len(actual)} records; expected {len(expected)}. Resume the benchmark before plotting final results.')
    references={pid:read(root/'references'/f'{pid}.json')['new_token_ids'] for pid in plan['ids']['test']}
    for row in rows:
        tokens=references[row['pubid']]
        if row['output_sha256']!=hashlib.sha256(json.dumps(tokens).encode()).hexdigest() or row['tokens']!=len(tokens):
            raise ValueError('Output identity/count validation failed')
        import math
        if not math.isfinite(row['seconds']) or row['seconds']<=0:
            raise ValueError('Invalid timing')
    training=[]
    for seed in cfg['policy_seeds']:
        for index,pid in enumerate(plan['ids']['train']):
            row=read(root/'training_blocks'/f'{seed}_{index:05d}.json')['row']
            if row['pubid']!=pid or row['method']!=f'adaptive:{seed}':
                raise ValueError('Training checkpoint mismatch')
            training.append(row)
    return plan,methods,frozen,online,rows,training


def analyze(directory):
    root=Path(directory)
    plan,methods,frozen,online,rows,training=load_complete(root)
    cfg=plan['settings']
    from benchmark_pubmedqa import summarize,paired_speedup
    summary=summarize(rows)
    for stats in summary.values(): stats.pop('peak_allocated_mib',None)
    baselines=[m for m in methods if m not in frozen+online]
    comparisons={name:{base:paired_speedup(rows,base,name,seed=2718,samples=5000) for base in baselines} for name in frozen}
    training_seconds={name:sum(r['seconds'] for r in training if r['method']==name) for name in frozen}
    metadata=dict(stage='final' if not cfg['smoke'] else 'SMOKE ONLY',test_count=len(plan['ids']['test']),
                  train_count=len(plan['ids']['train']),repeats=cfg['repeats'],max_new_tokens=cfg['max_new_tokens'],
                  policy_seeds=cfg['policy_seeds'],epsilon=cfg['epsilon'],configs=plan['configs'])
    data=dict(metadata=metadata,methods={name:summary[name] for name in methods},comparisons=comparisons,
              training_seconds=training_seconds,all_outputs_match=True,
              online_comparisons={name:{base:dict(speedup=summary[base]['seconds']/summary[name]['seconds']) for base in baselines} for name in online},
              action_counts={name:dict(Counter(str(tuple(s['k_config'])) for row in rows if row['method']==name for s in row['trace'])) for name in frozen+online})
    (root/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    for filename,records in [('test.jsonl',rows),('training.jsonl',training)]:
        with (root/filename).open('w',encoding='utf-8') as f:
            for row in records: f.write(json.dumps(row)+'\n')
    with (root/'results.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['method','tokens_per_second','seconds','tokens','calls','speedup_vs_fixed','throughput_improvement_percent','latency_reduction_percent','ci_low','ci_high'])
        for name in methods:
            stats=summary[name]
            speed=summary['fixed:2,2,1']['seconds']/stats['seconds']
            interval=comparisons.get(name,{}).get('fixed:2,2,1',{}).get('bootstrap_95_percent',['',''])
            writer.writerow([name,stats['tokens_per_second'],stats['seconds'],stats['tokens'],stats['calls'],speed,100*(speed-1),100*(1-1/speed),*interval])
    totals=defaultdict(float)
    for row in rows: totals[row['pubid'],row['method']]+=row['seconds']
    with (root/'per_question.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['pubid','method','seconds_across_repeats','speedup_vs_fixed'])
        for pid in plan['ids']['test']:
            for name in methods: writer.writerow([pid,name,totals[pid,name],totals[pid,'fixed:2,2,1']/totals[pid,name]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def save(fig,name):
        for suffix in ('png','svg','pdf'): fig.savefig(root/f'{name}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    labels={name:name.replace('adaptive:','Frozen seed ').replace('online:','Online seed ') for name in methods}
    labels.update({'target':'Target only','fixed:2,2,1':'Fixed (2,2,1)',
                   'validation_best_branching':f'Pilot static {tuple(plan["static_reference"])}'})
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    colors=['#31966b' if n in frozen else '#d78c35' if n in online else '#507fa6' for n in methods]
    bars=ax.barh([labels[n] for n in methods],[summary[n]['tokens_per_second'] for n in methods],color=colors)
    ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.invert_yaxis()
    ax.set(xlabel='Generated tokens / second',title=f'{metadata["stage"]}: {metadata["test_count"]} questions, {cfg["repeats"]} repeats, {cfg["max_new_tokens"]}-token cap')
    ax.set_xlim(0,max(s['tokens_per_second'] for s in summary.values())*1.15)
    save(fig,'throughput')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    alternate='validation_best_branching' if 'validation_best_branching' in methods else 'target'
    for ax,base in zip(axes,['fixed:2,2,1',alternate]):
        for i,name in enumerate(frozen+online):
            speed=summary[base]['seconds']/summary[name]['seconds']
            color='#31966b' if name in frozen else '#d78c35'
            if name in frozen:
                interval=comparisons[name][base]['bootstrap_95_percent']
                ax.plot([100*(x-1) for x in interval],[i,i],color=color,lw=2)
            ax.scatter([100*(speed-1)],[i],color=color,marker='o' if name in frozen else 'D')
        ax.axvline(0,color='black',ls='--')
        ax.set_yticks(range(len(frozen+online)),[labels[n] for n in frozen+online])
        ax.invert_yaxis()
        ax.set(xlabel='Throughput change (%) — negative means slower',title=f'Against {labels[base]}')
    fig.suptitle('Frozen: 95% paired question-bootstrap intervals; online: descriptive points')
    save(fig,'improvement')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in frozen:
        speeds=sorted(totals[pid,'fixed:2,2,1']/totals[pid,name] for pid in plan['ids']['test'])
        axes[0].plot(speeds,np.arange(1,len(speeds)+1)/len(speeds),label=labels[name])
    axes[0].axvline(1,color='black',ls='--')
    axes[0].set(xlabel='Per-question speedup vs fixed',ylabel='Fraction of test questions',title='Distribution of gains and regressions')
    repeat_rows=[]
    for name in frozen+online:
        values=[]
        for repeat in range(cfg['repeats']):
            elapsed=sum(r['seconds'] for r in rows if r['repeat']==repeat and r['method']==name)
            baseline=sum(r['seconds'] for r in rows if r['repeat']==repeat and r['method']=='fixed:2,2,1')
            values.append(baseline/elapsed)
            repeat_rows.append([name,repeat+1,baseline/elapsed])
        axes[1].plot(range(1,cfg['repeats']+1),values,'o-',label=labels[name])
    axes[1].axhline(1,color='black',ls='--')
    axes[1].set(xlabel='Repeat / reset online stream',ylabel='Aggregate speedup vs fixed',xticks=range(1,cfg['repeats']+1),title='Repeat and stream-order variation')
    for ax in axes: ax.legend(fontsize=8)
    save(fig,'variation')
    with (root/'repeat_results.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['method','repeat','speedup_vs_fixed']); writer.writerows(repeat_rows)
    configs=[str(tuple(c)) for c in plan['configs']]
    counts=np.array([[data['action_counts'][name].get(c,0) for c in configs] for name in frozen+online])
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
    im=ax.imshow(counts/counts.sum(axis=1,keepdims=True),aspect='auto',cmap='Blues',vmin=0)
    ax.set_xticks(range(len(configs)),configs,rotation=65,ha='right',fontsize=8)
    ax.set_yticks(range(len(frozen+online)),[labels[n] for n in frozen+online])
    ax.set(title='Tree selection frequency',xlabel='Branching widths at successive depths')
    fig.colorbar(im,ax=ax,label='Fraction of decoding steps')
    save(fig,'action_usage')
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name in frozen:
        group=[r for r in training if r['method']==name]
        blocks=[group[i:i+20] for i in range(0,len(group),20)]
        ax.plot(np.cumsum([len(b) for b in blocks]),[sum(r['tokens'] for r in b)/sum(r['seconds'] for r in b) for b in blocks],'o-',label=labels[name])
    ax.set(xlabel='Training questions processed',ylabel='Tokens / second',title='Training throughput; prompt difficulty varies between blocks')
    ax.legend()
    save(fig,'training_throughput')
    lines=['# Final PubMedQA results' if not cfg['smoke'] else '# SMOKE TEST — NOT PAPER RESULTS','',
           f'Validated {len(rows):,} timed test calls: {metadata["test_count"]} questions × {len(methods)} methods × {cfg["repeats"]} repeats. '
           f'Training: {metadata["train_count"]} development questions per seed. All test tokens match target-only greedy outputs.', '',
           '| Method | Tokens/s | Speedup vs fixed | Throughput change | 95% interval (speedup) |',
           '|---|---:|---:|---:|---|']
    for name in methods:
        stats=summary[name]
        speed=summary['fixed:2,2,1']['seconds']/stats['seconds']
        interval=comparisons.get(name,{}).get('fixed:2,2,1',{}).get('bootstrap_95_percent')
        text=f'[{interval[0]:.4f}, {interval[1]:.4f}]' if interval else 'descriptive' if name in online else '—'
        lines.append(f'| {labels[name]} | {stats["tokens_per_second"]:.2f} | {speed:.4f}× | {100*(speed-1):+.2f}% | {text} |')
    lines+=['','## Interpretation','']
    for name in frozen:
        for base in ['fixed:2,2,1',alternate]:
            c=comparisons[name][base]; low,high=c['bootstrap_95_percent']
            conclusion='supports a speed gain' if low>1 else 'supports a slowdown' if high<1 else 'does not establish a difference from 1'
            lines.append(f'- {labels[name]} vs {labels[base]}: {c["speedup"]:.4f}× [{low:.4f}, {high:.4f}]; this interval {conclusion}.')
    lines+=['','Intervals use 5,000 paired bootstrap resamples of whole questions, preserving all repeats. '
            'They are conditional on these trained policies and this GPU/runtime, not confidence intervals over all possible training seeds. '
            'Online points include exploration, step timing and updates. Online repeats reset their policy and change question order; they are dependent streams, so no independent-question interval is assigned.', '',
            '## Training and measurement costs','',
            *[f'- {labels[name]}: {seconds:.1f} seconds of training decoding (excluded from test speedups).' for name,seconds in training_seconds.items()], '',
            'Model loading, correctness-reference generation, checkpoint writes, tokenization and final text decoding are excluded from timed decoding. '
            'Prefill, tree construction, verification and controller work are included. No configuration sweep was performed. '
            'These training times measure all training decoding, not incremental overhead over useful generation. '
            'The pilot-selected static tree is a reference, not a proven globally optimal static tree.', '',
            '## Scope','',
            f'The locked menu contains {len(plan["configs"])} branching shapes. Fixed baseline: {tuple(plan["fixed_config"])}. '
            f'Epsilon: {cfg["epsilon"]}; untried actions within a context are explored first, so initial exploration can exceed epsilon. '
            f'Output cap: {cfg["max_new_tokens"]}, prompt cap: {cfg["max_prompt_tokens"]}, greedy decoding, batch size one, FP32. '
            'Models: cached DistilGPT2 draft and GPT2-large target. This evaluates decoding speed on biomedical prompts, not medical-answer accuracy.', '',
            'Test questions were excluded from previously saved prompt artifacts at plan preparation (except in explicitly labeled smoke mode). '
            'Previously evaluated questions may be used as development/training data; they are not counted as fresh test observations. '
            'Freshness is relative to this workspace artifact audit, not proof against model-pretraining exposure or unrecorded experiments. '
            'Do not tune settings after seeing this test and still describe it as untouched final evaluation.', '',
            '## Figures','',
            'All figures have PNG, SVG and PDF versions. `improvement` is the main paper comparison; '
            '`throughput` gives absolute speed; `variation` shows per-question and repeat variability; '
            '`action_usage` shows selected trees; `training_throughput` is descriptive, not a controlled learning curve.', '',
            'Use `results.csv`, `per_question.csv`, `repeat_results.csv`, `summary.json`, and the exported JSONL files for further analysis. '
            'Speedup = total baseline time / total method time for matched outputs. Throughput change = 100 × (speedup − 1); '
            'latency reduction = 100 × (1 − 1/speedup). They are different percentages.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Validated and plotted {len(rows):,} calls. Read {root/"RESULTS.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',default='pubmedqa_final')
    analyze(parser.parse_args().directory)
