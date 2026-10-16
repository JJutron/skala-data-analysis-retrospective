"""Train-only nested chronological selection and causal two-target inference.

No evaluation labels, prior selected models, or prior submission predictions are read.
All outputs live in clean_causal_run; existing project artifacts are preserved.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from features import augment_features, MOMENTUM_FEATURES_TARGET_1, OPTIMIZED_FEATURES_TARGET_1
from validate import make_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'clean_causal_run'
TARGETS = ['Target_1', 'Target_2']
FIRST = 28
GAP = 3
OUTER = [(474, 150), (624, 150), (774, 150)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data(path):
    # Explicit allowlist also applies during --verify.
    assert Path(path).resolve().parent == ROOT
    assert Path(path).name in {
        'train.csv', 'test_public.csv', 'test_private.csv',
        'sample_submission_public.csv', 'sample_submission_private.csv'}
    raw = pd.read_csv(path, encoding='cp949')
    assert raw.Time_step.is_unique and raw.Time_step.notna().all()
    assert raw.Time_step.diff().dropna().eq(1).all()
    return raw


def feature_frames(raw):
    inputs = raw.drop(columns=['Time_step', *TARGETS], errors='ignore').copy()
    assert inputs.Weekday.between(1, 7).all()
    day = inputs.Weekday.astype(int)
    extra = {f'day{k}': day.eq(k).astype(float) for k in range(1, 8)}
    fi = [c for c in inputs if c.startswith('FI_')]
    for col in fi:
        change = inputs[col].diff()
        volatility = change.shift(1).rolling(28, min_periods=7).std().clip(lower=1e-8)
        for lag in [1, 3, 7, 14]:
            extra[f'{col}_change{lag}'] = inputs[col].diff(lag)
            extra[f'{col}_shock{lag}'] = inputs[col].diff(lag) / volatility
        if col in ['FI_Sp500', 'FI_Nasdaq', 'FI_Dow']:
            for lag in [0, 1, 2, 3]:
                extra[f'{col}_lagchange{lag}'] = change.shift(lag)
            for k in range(1, 8):
                extra[f'{col}_daychange{k}'] = change * day.eq(k)
    shops = [c for c in inputs if c.startswith(('SH1_', 'SH2_'))
             and ('식품' in c or '생활건강' in c)]
    # D+3 WEEKDAY is known from the calendar. Demand at D+3 is not observed.
    future_day = (day + 2) % 7 + 1
    for col in shops:
        s = inputs[col]
        history = {k: s.loc[day.eq(k)].ewm(alpha=.25, adjust=False).mean()
                   .reindex(raw.index).ffill().shift(1) for k in range(1, 8)}
        forecast = pd.Series(np.nan, index=raw.index)
        for k in range(1, 8):
            mask = future_day.eq(k)
            forecast.loc[mask] = history[k].loc[mask]
        extra[f'{col}_calendar_delta'] = forecast - s
        extra[f'{col}_calendar_ratio'] = forecast / (s.abs() + .005)
        extra[f'{col}_trend7'] = s.diff(7)
        # D+3 shares its weekday with D-4, D-11, D-18 and D-25.
        expected = pd.concat([s.shift(k) for k in [4, 11, 18, 25]], axis=1).mean(axis=1)
        extra[f'{col}_season_ratio'] = expected / (s.abs() + .005)
    for col in [c for c in inputs if c.startswith('KE_') and c != 'KE_QR체크인']:
        past = inputs[col].shift(1).rolling(28, min_periods=7)
        extra[f'{col}_z28'] = (inputs[col] - past.mean()) / past.std().clip(lower=.005)
    augmented = augment_features(inputs).drop(columns='Weekday')
    frame = pd.concat([augmented, pd.DataFrame(extra, index=raw.index)], axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame['Weekday'] = day.astype(str)
    days = [f'day{k}' for k in range(1, 8)]
    groups = {
        'momentum': list(MOMENTUM_FEATURES_TARGET_1),
        'core': [c for c in OPTIMIZED_FEATURES_TARGET_1 if c != 'Weekday'] + days,
        'macro': [f'{c}_change{k}' for c in fi for k in [1, 3, 7, 14]] + days,
        'shocks': [f'{c}_shock{k}' for c in fi for k in [1, 3, 7, 14]] + days,
        'us_calendar': [c for c in extra if '_lagchange' in c or '_daychange' in c] + days,
        'raw': list(inputs.columns),
        'food_calendar': shops + [f'{c}_{suffix}' for c in shops for suffix in
                                 ['calendar_delta', 'calendar_ratio', 'trend7', 'season_ratio']] + ['Weekday'],
        'food_ratios': [c for c in extra if '_season_ratio' in c or '_calendar_ratio' in c or '_z28' in c] + days,
    }
    assert all(set(cols).isdisjoint(TARGETS + ['Time_step']) for cols in groups.values())
    return frame, groups


def specifications(target):
    if target == 'Target_1':
        specs = [dict(name='v2_baseline', kind='baseline_fin', group='momentum')]
        specs += [dict(name=f'{g}_lr_{c}', kind='lr', group=g, C=c)
                  for g in ['core', 'shocks', 'us_calendar'] for c in [.003, .03]]
        specs += [dict(name='core_lda', kind='lda', group='core'),
                  dict(name='macro_hist', kind='hist', group='macro'),
                  dict(name='core_nca', kind='nca', group='core')]
    else:
        specs = [dict(name='v2_baseline', kind='baseline_food', group='raw'),
                 dict(name='calendar_cat3', kind='cat', group='food_calendar', depth=3),
                 dict(name='calendar_cat5', kind='cat', group='food_calendar', depth=5)]
        specs += [dict(name=f'ratios_lr_{c}', kind='lr', group='food_ratios', C=c) for c in [.01, .1]]
    return specs


def estimator(spec, cols):
    kind = spec['kind']
    if kind == 'baseline_fin':
        return make_model('momentum', cols, [c for c in cols if c != 'Weekday'], ['Weekday'])
    if kind in ['cat', 'baseline_food']:
        return CatBoostClassifier(iterations=300, depth=spec.get('depth', 4),
            learning_rate=.03, l2_leaf_reg=3 if kind == 'baseline_food' else 10,
            cat_features=['Weekday'], random_seed=42, thread_count=2,
            verbose=False, allow_writing_files=False)
    if kind == 'lr':
        m = LogisticRegression(C=spec['C'], max_iter=2000, random_state=42)
    elif kind == 'lda':
        m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.8)
    elif kind == 'nca':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
            NeighborhoodComponentsAnalysis(n_components=2, max_iter=60, init='pca', random_state=42),
            KNeighborsClassifier(n_neighbors=45, weights='distance'))
    else:
        m = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=7,
            learning_rate=.03, l2_regularization=10, min_samples_leaf=25, random_state=42)
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), RobustScaler(), m)


def design(frame, cols, spec):
    x = frame[cols]
    # Fixed missing-value placeholder; no test-dependent statistics for CatBoost.
    return x.fillna(0) if spec['kind'] in ['cat', 'baseline_food'] else x


def fit_one(frame, groups, labels, spec, end):
    cols = groups[spec['group']]
    assert FIRST < end - GAP
    m = estimator(spec, cols)
    with warnings.catch_warnings(record=True) as caught:
        m.fit(design(frame.iloc[FIRST:end-GAP], cols, spec), labels.iloc[FIRST:end-GAP])
    return dict(model=m, spec=spec, features=cols, fit_first_time=FIRST+1,
                fit_last_time=end-GAP, label_maturity_last_time=end,
                warnings=[str(w.message) for w in caught])


def predict_one(bundle, frame):
    m = bundle['model']
    return m.predict_proba(design(frame, bundle['features'], bundle['spec']))[:, list(m.classes_).index(1)]


def selection(raw, target, folds, cache):
    # This function receives ONLY the prefix before the outer evaluation.
    frame, groups = feature_frames(raw)
    assert max(end+size for end, size in folds) <= len(raw)-GAP
    specs = specifications(target)
    records = []
    predictions = {}
    for spec in specs:
        per_fold = []
        for end, size in folds:
            key = (target, spec['name'], end, size)
            if key not in cache:
                b = fit_one(frame, groups, raw[target], spec, end)
                p = predict_one(b, frame.iloc[end:end+size])
                cache[key] = (p, b['warnings'])
            per_fold.append(cache[key][0])
        scores = [float(roc_auc_score(raw[target].iloc[e:e+s], p)) for (e,s),p in zip(folds, per_fold)]
        records.append(dict(spec=spec, scores=scores, mean=float(np.mean(scores)),
                            warnings=[cache[(target,spec['name'],e,s)][1] for e,s in folds]))
        predictions[spec['name']] = per_fold
    best = max(records, key=lambda r:r['mean'])
    diverse = []
    for r in sorted(records, key=lambda r:r['mean'], reverse=True):
        family = 'lr' if r['spec']['kind'] == 'baseline_fin' else r['spec']['kind']
        if family not in [f for f,_ in diverse]:
            diverse.append((family,r))
        if len(diverse) == 3:
            break
    members = [r['spec'] for _,r in diverse]
    blend = [np.mean([predictions[s['name']][i] for s in members], axis=0) for i in range(len(folds))]
    blend_scores = [float(roc_auc_score(raw[target].iloc[e:e+s],p)) for (e,s),p in zip(folds,blend)]
    # Freeze this gate in code before scoring outer windows or any test data.
    selected = members if np.mean(blend_scores) > best['mean']+.002 else [best['spec']]
    return dict(records=records, selected=selected, blend_scores=blend_scores,
                selection_label_last_time=int(raw.Time_step.iloc[max(e+s for e,s in folds)-1]))


def fit_recipe(raw, target, specs):
    frame, groups = feature_frames(raw)
    return [fit_one(frame, groups, raw[target], s, len(raw)) for s in specs]


def predict_recipe(bundles, raw):
    frame, _ = feature_frames(raw)
    return np.mean([predict_one(b, frame) for b in bundles], axis=0)


def causality_checks(raw, bundles):
    # Check EACH evaluation date by removing every observation after that date.
    full = {t:predict_recipe(bundles[t], raw) for t in TARGETS}
    for index in range(924, len(raw)):
        prefix = raw.iloc[:index+1]
        frame, _ = feature_frames(prefix)
        for t in TARGETS:
            p = np.mean([predict_one(b, frame.iloc[[-1]])[0] for b in bundles[t]])
            np.testing.assert_allclose(p, full[t][index], rtol=0, atol=1e-12)
    changed = raw.copy()
    changed[TARGETS] = 1
    for t in TARGETS:
        np.testing.assert_allclose(full[t], predict_recipe(bundles[t], changed), rtol=0, atol=1e-12)
    return full


def verify_preservation():
    snapshot = json.loads((OUT/'existing_hashes.json').read_text())
    for path, expected in snapshot.items():
        assert sha(ROOT/path) == expected, f'Existing artifact modified: {path}'
    return len(snapshot)


def export(bundles, verify=False):
    train = read_data(ROOT/'train.csv')
    checks = {}
    for kind in ['public', 'private']:
        data = read_data(ROOT/f'test_{kind}.csv').drop(columns=TARGETS, errors='ignore')
        sample = read_data(ROOT/f'sample_submission_{kind}.csv')
        assert data.Time_step.min() == train.Time_step.max()+1
        assert set(data.Time_step) == set(sample.Time_step)
        raw = pd.concat([train, data], ignore_index=True)
        all_predictions = causality_checks(raw, bundles)
        out = sample.copy()
        for target,col in zip(TARGETS,['pred_1','pred_2']):
            by_time = pd.Series(all_predictions[target], index=raw.Time_step)
            out[col] = by_time.loc[sample.Time_step].to_numpy()
        assert out.shape == (100 if kind=='public' else 200, 3)
        assert list(out) == ['Time_step','pred_1','pred_2']
        p = out[['pred_1','pred_2']].to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        path = OUT/f'{kind}_causal_clean.csv'
        if verify:
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        else:
            assert not path.exists()
            out.to_csv(path,index=False)
            pd.testing.assert_frame_equal(out,pd.read_csv(path),rtol=0,atol=1e-12)
        checks[kind] = dict(rows=len(out), all_rows_prefix_invariant=True,
                            targets_ignored=True, sha256=sha(path))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        manifest = json.loads((OUT/'report.json').read_text())
        for path,expected in manifest['source_hashes'].items():
            assert sha(ROOT/path) == expected
        for target,expected in manifest['model_hashes'].items():
            assert sha(OUT/f'{target}.joblib') == expected
        bundles = {t:joblib.load(OUT/f'{t}.joblib') for t in TARGETS}
        checks = export(bundles, verify=True)
        assert checks == manifest['inference_checks']
        print('PASS: both saved models, all 300 prediction prefixes, labels ignored, CSV reproduction,',
              verify_preservation(), 'existing artifacts unchanged.')
        return
    assert not (OUT/'report.json').exists()
    raw = read_data(ROOT/'train.csv')
    assert len(raw)==924 and all(set(raw[t].unique())=={0,1} for t in TARGETS)
    frame,_ = feature_frames(raw)
    pd.testing.assert_frame_equal(frame.iloc[:500], feature_frames(raw.iloc[:500])[0])
    cache = {}
    report = dict(state='training', gap_rows=GAP, first_train_time=FIRST+1,
        selection_uses_public_labels=False, uses_future_observations=False,
        previous_models_or_predictions_used=False, outer={}, final={},
        note='Train-only nested chronological development validation. These historical labels were used in prior work; this is not a fresh independent holdout. D+3 is provisionally three rows. Current-day features require availability at the prediction cutoff.')
    outer_predictions = []
    for target in TARGETS:
        report['outer'][target] = []
        for end,size in OUTER:
            prefix = raw.iloc[:end].copy()
            inner = [(end-203,100),(end-103,100)]
            choice = selection(prefix, target, inner, cache)
            bundles = fit_recipe(prefix,target,choice['selected'])
            baseline = fit_recipe(prefix,target,[specifications(target)[0]])
            selected_p = predict_recipe(bundles,raw.iloc[:end+size])[end:]
            base_p = predict_recipe(baseline,raw.iloc[:end+size])[end:]
            y = raw[target].iloc[end:end+size]
            scores = dict(selected=float(roc_auc_score(y,selected_p)),baseline=float(roc_auc_score(y,base_p)))
            assert choice['selection_label_last_time']+GAP < int(raw.Time_step.iloc[end])
            record = dict(validation_first_time=end+1,validation_last_time=end+size,
                          fit_last_time=end-GAP,selection=choice,**scores)
            report['outer'][target].append(record)
            outer_predictions.extend(dict(target=target,Time_step=int(time),label=int(label),
                                          selected=float(a),baseline=float(b))
                                     for time,label,a,b in zip(raw.Time_step.iloc[end:end+size],y,selected_p,base_p))
            print('OUTER',target,end,scores,'selected',[s['name'] for s in choice['selected']],flush=True)
            (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # All final settings are chosen within train, before opening test inputs.
    bundles = {}
    for target in TARGETS:
        choice = selection(raw,target,[(521,100),(621,100),(721,100),(821,100)],cache)
        report['final'][target] = choice
        bundles[target] = fit_recipe(raw,target,choice['selected'])
        joblib.dump(bundles[target],OUT/f'{target}.joblib')
        print('FINAL',target,[s['name'] for s in choice['selected']],flush=True)
    report['outer_mean_auc'] = {k:float(np.mean([r[k] for records in report['outer'].values() for r in records]))
                                for k in ['baseline','selected']}
    report['outer_target_mean_auc'] = {t:{k:float(np.mean([r[k] for r in report['outer'][t]]))
                                            for k in ['baseline','selected']} for t in TARGETS}
    report['outer_improvement'] = report['outer_mean_auc']['selected']-report['outer_mean_auc']['baseline']
    report['final_fit_last_time'] = 921
    report['public_auc'] = None
    report['private_auc'] = None
    report['source_hashes'] = {p:sha(ROOT/p) for p in ['clean_causal.py','features.py','validate.py',
        'train.csv','test_public.csv','test_private.csv','sample_submission_public.csv','sample_submission_private.csv']}
    report['model_hashes'] = {t:sha(OUT/f'{t}.joblib') for t in TARGETS}
    pd.DataFrame(outer_predictions).to_csv(OUT/'nested_predictions.csv',index=False)
    # Save frozen choices before inference; test files cannot influence selection.
    report['state'] = 'recipes_frozen'
    (OUT/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    report['inference_checks'] = export(bundles)
    report['existing_artifacts_preserved'] = verify_preservation()
    report['state'] = 'completed'
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('RESULT',report['outer_target_mean_auc'],report['outer_mean_auc'],flush=True)


if __name__ == '__main__':
    main()
