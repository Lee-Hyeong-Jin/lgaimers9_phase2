ID_COL, TARGET_COL = 'row_id', 'control_success'


def bayes_init(df, F, stats, meta):
    """Row-wise dynamic-Bayesian pitcher level (uses the row's own season-to-date record + training-season totals)."""
    off = str(meta.get('offset', ''))
    bay = bayes_features(df, F, stats, src=off.endswith('_src'), **BAYES_HP)
    mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mx = max(mu)
    mu_row = pd.Series(np.asarray(df.season)).map(mu).fillna(mu[mx]).values
    pm = np.clip(mu_row + bay['bayes_dev'] + ((F['CTX_hand_adj'].values + F['CTX_2k_adj'].values) if off.startswith('bayes_ctx') else 0.0), 0.05, 0.95)
    if meta.get('type') in ('lgb_mc', 'cat_mc'):
        fs = meta['fshare']
        return np.stack([np.log(pm)] + [np.log((1 - pm) * fs[k]) for k in range(3)], 1)
    return logit(pm)


def main():
    TEST_DIR, MODEL_DIR, OUT_DIR = './data', './model', './output'
    test = pd.read_csv(os.path.join(TEST_DIR, 'test.csv'), encoding='utf-8-sig', dtype={'base_state': str, 'top_bottom': str, 'game_type': str, 'row_id': str})  # keep code columns as strings regardless of file size
    sub = pd.read_csv(os.path.join(TEST_DIR, 'sample_submission.csv'), encoding='utf-8-sig')
    print('test', test.shape, 'sub', sub.shape, flush=True)
    stats = joblib.load(os.path.join(MODEL_DIR, 'stats.pkl'))
    ens = json.load(open(os.path.join(MODEL_DIR, 'ensemble.json')))
    preds = []
    weights = []
    feat_cache = {}
    for mfile in ens['models']:
        meta = json.load(open(os.path.join(MODEL_DIR, mfile)))
        mode = meta.get('mode', 'prev'); guess = meta.get('level_guess')
        key = (mode, json.dumps(guess, sort_keys=True))
        if key not in feat_cache:
            # Row-wise features: each row uses only its own columns + training-data statistics.
            F = add_change_features(build_features(test, stats, mode=mode, guess=guess))
            feat_cache[key] = F
        F = feat_cache[key]
        if meta.get('type') == 'nn':
            import torch
            prep = joblib.load(os.path.join(MODEL_DIR, meta['prep']))
            Fn = F.copy(); Fn['season_rel'] = np.clip(prep['max_season'] - Fn['season'].values, 0, 10)
            Xn = transform_numeric(Fn, prep['num_feats'], prep['qt'], prep['nan_cols'])
            Xc = transform_cats(Fn, prep['cats'], prep['maps'])
            nets = []
            for b in meta['boosters']:
                net = build_net(Xn.shape[1], prep['sizes'], tuple(prep['hidden']), prep['drop'])
                net.load_state_dict(torch.load(os.path.join(MODEL_DIR, b), map_location='cpu')); nets.append(net)
            init4 = bayes_init(test, F, stats, {**meta, 'type': 'lgb_mc'}) if str(meta.get('offset', '')).startswith('bayes') else None
            pn = nn_predict(nets, Xn, Xc, init4=init4)
            raw = logit(np.clip(pn, 1e-6, 1 - 1e-6)) + float(meta.get('logit_shift', 0.0))
            preds.append(expit(raw)); weights.append(float(meta.get('weight', 1.0)))
            print('model', mfile, 'pred mean', round(float(preds[-1].mean()), 4), flush=True)
            continue
        X = F[meta['feats']]
        raw = np.zeros(len(X))
        init = None
        if str(meta.get('offset', '')).startswith('bayes'):
            init = bayes_init(test, F, stats, meta)
        if meta.get('type') in ('cat', 'cat_mc'):
            from catboost import CatBoostClassifier
            Xc = X.copy()
            for c in meta['cat']: Xc[c] = Xc[c].astype(int)
            for b in meta['boosters']:
                m = CatBoostClassifier(); m.load_model(os.path.join(MODEL_DIR, b))
                if init is not None:
                    from catboost import Pool
                    rw = m.predict(Pool(Xc, cat_features=meta['cat'], baseline=init), prediction_type='RawFormulaVal')
                    pr = softmax(rw, axis=1)[:, 0] if meta.get('type') == 'cat_mc' else expit(rw)
                else:
                    pr = m.predict_proba(Xc); pr = pr[:, 0] if meta.get('type') == 'cat_mc' else pr[:, 1]
                raw += logit(np.clip(pr, 1e-6, 1 - 1e-6))
            raw /= len(meta['boosters']); raw += float(meta.get('logit_shift', 0.0))
            preds.append(expit(raw)); weights.append(float(meta.get('weight', 1.0)))
            print('model', mfile, 'pred mean', round(float(preds[-1].mean()), 4), flush=True)
            continue
        for b in meta['boosters']:
            m = lgb.Booster(model_file=os.path.join(MODEL_DIR, b))
            if meta.get('type') == 'lgb_mc':
                if init is not None:
                    pr = softmax(m.predict(X, raw_score=True) + init, axis=1)[:, 0]
                else:
                    pr = m.predict(X)[:, 0]
                raw += logit(np.clip(pr, 1e-6, 1 - 1e-6))
            else:
                raw += m.predict(X, raw_score=True) + (0.0 if init is None else init)
        raw /= len(meta['boosters'])
        if meta.get('offset'):
            raw += logit(F['L_ref'].values)
        raw += float(meta.get('logit_shift', 0.0))
        preds.append(expit(raw)); weights.append(float(meta.get('weight', 1.0)))
        print('model', mfile, 'pred mean', round(float(preds[-1].mean()), 4), flush=True)
    w = np.array(weights) / np.sum(weights)
    if ens.get('blend', 'prob') == 'logit':
        p = expit(sum(wi * logit(np.clip(pi, 1e-6, 1 - 1e-6)) for wi, pi in zip(w, preds)))
    else:
        p = sum(wi * pi for wi, pi in zip(w, preds))
    p = p + float(ens.get('prob_shift', 0.0))
    # label-source-level adjustments using ONLY the row's own columns (team ids, game type, top/bottom, count); each block independent
    is_f = (test['game_type'].astype(str).values == 'F')
    is_lg = ((test['pitcher_team_id'].values == 13) | (test['batter_team_id'].values == 13)) & (~is_f)
    is_lgp = is_lg & (test['pitcher_team_id'].values == 13)
    lg_p = float(ens.get('lg_shift_p', ens.get('lg_shift', 0.0)) or 0.0); lg_o = float(ens.get('lg_shift_opp', ens.get('lg_shift', 0.0)) or 0.0); f_s = float(ens.get('f_shift', 0.0) or 0.0)
    p = p + np.where(is_lgp, lg_p, 0.0) + np.where(is_lg & ~is_lgp, lg_o, 0.0) + np.where(is_f, f_s, 0.0)
    lg_home = is_lg & (((test['top_bottom'].astype(str).values == 'T') & (test['pitcher_team_id'].values == 13)) | ((test['top_bottom'].astype(str).values == 'B') & (test['batter_team_id'].values == 13)))
    p = p + np.where(lg_home, float(ens.get('lg_shift_home', 0.0) or 0.0), 0.0) + np.where(is_lg & ~lg_home, float(ens.get('lg_shift_away', 0.0) or 0.0), 0.0)
    is_30 = (test['balls_before'].values == 3) & (test['strikes_before'].values == 0)
    p = p + np.where(is_lg & is_30, float(ens.get('lg30_shift', 0.0) or 0.0), 0.0)
    if ens.get('cold_slope') or ens.get('sharpen'):
        # Row-wise sharpening for under-confident segments defined by the row's OWN features; center is a training-data constant.
        center = logit(float(ens.get('cold_center', 0.486)))
        lastF = list(feat_cache.values())[-1]
        rules = list(ens.get('sharpen', []))
        if ens.get('cold_slope'): rules.append({'col': 'p_prior_n', 'lo': -1, 'hi': 0, 'slope': float(ens['cold_slope'])})
        lpp = logit(np.clip(p, 1e-6, 1 - 1e-6))
        for rule in rules:
            v = lastF[rule['col']].values
            mask = (v > rule['lo']) & (v <= rule['hi'])
            lpp[mask] = center + float(rule['slope']) * (lpp[mask] - center)
        p = expit(lpp)
    p = np.clip(p, 0.001, 0.999)
    pred_map = dict(zip(test[ID_COL].values, p))
    out = sub[[ID_COL]].copy()
    fill_const = float(stats['L_all'].loc[stats['L_all'].index.max()])  # training-data constant (no test-set dependence)
    out[TARGET_COL] = out[ID_COL].map(pred_map).fillna(fill_const).astype(float)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, 'submission.csv'), index=False, encoding='utf-8')
    print('saved', os.path.join(OUT_DIR, 'submission.csv'), out.shape, 'mean', round(float(out[TARGET_COL].mean()), 4))


if __name__ == '__main__':
    main()
