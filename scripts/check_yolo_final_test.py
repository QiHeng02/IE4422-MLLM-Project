import json, hashlib, sys
from pathlib import Path
from collections import defaultdict
root=Path('outputs/yolo/final_test')
m=json.loads((root/'manifest.json').read_text())
for name, expected in m['sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, 'Changed input: '+name
if sys.argv[1]=='before':
    for name in ('test','test_reworded'):
        assert not Path(f'results/predictions/yolo_rules_{name}.jsonl').exists(), 'Existing predictions'
    print('Frozen input hashes verified; no previous predictions', flush=True)
    sys.exit(0)
report={'input_hashes_verified':True,'splits':{}}
answers=defaultdict(list)
for name in ('test','test_reworded'):
    qs=[json.loads(x) for x in Path(f'data/processed/questions/{name}.jsonl').read_text().splitlines() if x.strip()]
    path=Path(f'results/predictions/yolo_rules_{name}.jsonl')
    ps=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    ids=[x['qid'] for x in ps]
    assert len(ids)==len(set(ids))==len(qs)
    assert set(ids)=={q['qid'] for q in qs}, 'Prediction qid mismatch'
    pred={p['qid']:p for p in ps}
    for q in qs:
        p=pred[q['qid']]
        assert p['conf']==m['conf']
        obj=json.loads(p['raw'])
        assert isinstance(obj['flagged'],list)
        answers[(q['image'],q['context'])].append(json.dumps(obj['flagged'],sort_keys=True))
    metric=Path(f'results/metrics/yolo_rules__{name}.json')
    scores=json.loads(metric.read_text())
    assert scores['n_questions']==len(qs) and scores['valid_json']==1.0
    report['splits'][name]={'questions':len(qs),'unique_qids':len(set(ids)),'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'metrics_sha256':hashlib.sha256(metric.read_bytes()).hexdigest(),'metrics':{k:scores.get(k) for k in ('known_precision','known_recall','known_f1','flag_f1','unseen_f1','box_precision','box_recall','rq3_precision','rq3_false_flag_rate')}}
report['canonical_paraphrase_identical']=all(len(set(v))==1 for v in answers.values())
(root/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
