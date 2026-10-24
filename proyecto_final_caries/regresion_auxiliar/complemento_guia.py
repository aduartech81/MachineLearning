"""Pruebas exigidas por la guía del Entregable 2 (secciones 5.2.2, 6.3, 6.4) sobre la regresión
auxiliar de glucemia ya ejecutada. No entrena: lee las predicciones OOF guardadas.

Uso (raíz del repo, con la corrida completa):
    python proyecto_final_caries/regresion_auxiliar/complemento_guia.py \
        --origen ../regresion_glucemia_20261023 --salida _static/regresion_glucemia_complemento
"""
from pathlib import Path
import argparse, json, math, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.tsa.stattools import bds
from statsmodels.graphics.tsaplots import plot_acf

SEED = 42
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031}  # Demšar (2006)

def p(x): return "<0,001" if x < 0.001 else f"{x:.3f}"

def holm(v):
    v = np.asarray(v, float); o = np.argsort(v); m = len(v); a = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * v[j]); a[j] = min(1, run)
    return a

def dm_hln(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2; n = len(d); s = d.mean() / math.sqrt(np.mean((d - d.mean()) ** 2) / n)
    s *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * stats.t.sf(abs(s), n - 1))

def cd_plot(r, cd, ruta, titulo):
    r = r.sort_values(); k = len(r); v = list(r.values)
    fig, ax = plt.subplots(figsize=(9, .55 * k + 1.6)); ax.set_xlim(.5, k + .5); ax.set_ylim(-.6, k + 1); ax.invert_xaxis(); ax.set_yticks([])
    for i, (m, x) in enumerate(r.items()): ax.plot(x, k - i, "o", color="#1f6f8b"); ax.text(x - .08, k - i, f"{m} ({x:.2f})", va="center", fontsize=9)
    lv = 0
    for i in range(k):
        j = max(jj for jj in range(i, k) if v[jj] - v[i] <= cd)
        if j > i: ax.plot([v[i], v[j]], [-.3 + .22 * lv] * 2, color="#c0392b", lw=3); lv += 1
    ax.plot([1, 1 + cd], [k + .6] * 2, "k", lw=2); ax.text(1 + cd / 2, k + .75, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.set_xlabel("Rango promedio de RMSE externo (1 = mejor)"); ax.set_title(titulo, fontsize=10)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ruta, dpi=150); plt.close(fig)

def main():
    a = argparse.ArgumentParser(); a.add_argument("--origen", type=Path, required=True); a.add_argument("--salida", type=Path, required=True)
    a = a.parse_args(); src, out = a.origen, a.salida; out.mkdir(parents=True, exist_ok=True)
    if (src / "metricas_por_fold.csv").exists():
        m = pd.read_csv(src / "metricas_por_fold.csv")
    else:  # la corrida final guarda un JSON por evaluación
        m = pd.DataFrame([json.loads(Path(x).read_text()) for x in sorted(glob.glob(str(src / "*_fold[0-9].json")))])
    assert len(m) == 140, f"Se esperaban 140 evaluaciones; hay {len(m)}"
    agg = m.groupby(["familia", "optimizador"]).agg(rmse=("rmse", "mean"), rmse_de=("rmse", "std"), mae=("mae", "mean"), mae_de=("mae", "std"), r2=("r2", "mean")).reset_index()
    best = agg.loc[agg.groupby("familia").rmse.idxmin()].sort_values("rmse").reset_index(drop=True)
    best.to_csv(out / "mejor_combinacion_por_familia.csv", index=False)
    # Friedman -> Nemenyi -> CD
    piv = pd.concat([m[(m.familia == f) & (m.optimizador == o)].set_index("fold").rmse.rename(f) for f, o in best[["familia", "optimizador"]].itertuples(index=False)], axis=1).sort_index()
    rk = piv.rank(axis=1).mean(); fr = stats.friedmanchisquare(*[piv[c] for c in piv]); k, N = piv.shape[1], piv.shape[0]
    cd = Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
    cd_plot(rk, cd, out / "cd_regresion.png", f"Glucemia: Friedman χ²={fr.statistic:.2f}, p={p(fr.pvalue)}; Nemenyi α=0,05")
    # OOF alineadas
    pred = {}
    for f, o in best[["familia", "optimizador"]].itertuples(index=False):
        d = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(str(src / f"{f}_{o}_fold*_pred.csv.gz")))]).sort_values("row_id").reset_index(drop=True)
        d["residuo"] = d.actual - d.prediccion; pred[f] = d
    ids = next(iter(pred.values())).row_id.to_numpy(); assert all(np.array_equal(d.row_id, ids) for d in pred.values())
    L = pd.DataFrame({f: d.residuo.to_numpy() ** 2 for f, d in pred.items()})
    # MCS
    mcs_txt = None
    try:
        from arch.bootstrap import MCS
        idx = np.sort(np.random.default_rng(SEED).choice(len(L), min(15000, len(L)), replace=False))
        mc = MCS(L.iloc[idx].reset_index(drop=True), size=.10, reps=1000, block_size=10, bootstrap="stationary", method="R", seed=SEED); mc.compute()
        t = mc.pvalues.copy(); t.columns = ["p_mcs"]; t["en_conjunto"] = t.index.isin(mc.included); t["perdida_media"] = L.mean().reindex(t.index)
        t.sort_values("perdida_media").to_csv(out / "mcs.csv"); mcs_txt = t.sort_values("perdida_media")
    except Exception as e:
        (out / "mcs_error.txt").write_text(repr(e))
    # Dirigidas
    b = best.familia.iloc[0]; rows = []
    for f in best.familia.iloc[1:]:
        e1, e2 = pred[b].residuo.to_numpy(), pred[f].residuo.to_numpy(); s, pd_ = dm_hln(e1, e2); d = e1 ** 2 - e2 ** 2
        g = np.random.default_rng(SEED); n = len(d)
        bt = [np.sqrt((e1[i] ** 2).mean()) - np.sqrt((e2[i] ** 2).mean()) for i in (g.integers(0, n, n) for _ in range(1000))]
        w = stats.wilcoxon(piv[b], piv[f], zero_method="zsplit"); df = piv[b] - piv[f]
        rows.append(dict(modelo_a=b, modelo_b=f, delta_rmse=float(np.sqrt((e1 ** 2).mean()) - np.sqrt((e2 ** 2).mean())), ic95_inf=float(np.percentile(bt, 2.5)), ic95_sup=float(np.percentile(bt, 97.5)),
                         dm_hln=s, p_dm=pd_, p_wilcoxon=float(w.pvalue), d_cohen_perdida=float(d.mean() / d.std(ddof=1)), d_cohen_folds=float(df.mean() / df.std(ddof=1))))
    c = pd.DataFrame(rows); c["p_dm_holm"] = holm(c.p_dm); c["p_wilcoxon_holm"] = holm(c.p_wilcoxon); c.to_csv(out / "comparaciones_dirigidas.csv", index=False)
    # Residuos
    dg = []
    for f, d in pred.items():
        r = d.residuo.to_numpy(); _, pw, _, _ = het_white(r, sm.add_constant(d.prediccion.to_numpy()))
        sub = np.sort(np.random.default_rng(SEED).choice(len(r), min(3000, len(r)), replace=False)); _, pb = bds(r[sub], max_dim=2)
        dg.append(dict(familia=f, p_white=float(pw), p_bds_m2=float(np.atleast_1d(pb)[0]), p_ljung_box_10=float(acorr_ljungbox(r, lags=[10], return_df=True).lb_pvalue.iloc[0]),
                       p_jarque_bera=float(stats.jarque_bera(r).pvalue)))
    dg = pd.DataFrame(dg); dg.to_csv(out / "supuestos_residuos.csv", index=False)
    r = pred[b].residuo; fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    plot_acf(r, lags=30, ax=ax[0]); ax[0].set(title=f"ACF de residuos OOF — {b} (orden de fila)", xlabel="Rezago")
    ax[1].hist(r, bins=80, density=True, alpha=.8); xs = np.linspace(r.quantile(.001), r.quantile(.999), 200); ax[1].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "k")
    ax[1].set(title="Histograma de residuos y normal de referencia", xlabel="Residuo (unidades del archivo)", ylabel="Densidad"); fig.tight_layout(); fig.savefig(out / "acf_histograma.png", dpi=150); plt.close(fig)
    # Markdown listo para el libro
    T = lambda df, cols, nom, dec=3: "| " + " | ".join(nom) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join((f"{x:.{dec}f}" if isinstance(x, float) else str(x)) for x in row) + " |" for row in df[cols].itertuples(index=False)) + "\n"
    dgs = dg.copy(); [dgs.__setitem__(col, dgs[col].map(p)) for col in ["p_white", "p_bds_m2", "p_ljung_box_10", "p_jarque_bera"]]
    cs = c.copy(); [cs.__setitem__(col, cs[col].map(p)) for col in ["p_dm", "p_dm_holm", "p_wilcoxon", "p_wilcoxon_holm"]]
    md = f"""### Pruebas exigidas por la guía (secciones 5.2.2, 6.3 y 6.4)

Se calcularon sobre las predicciones fuera de muestra ya guardadas, sin reentrenar. Los datos son transversales y no tienen orden temporal ni identificador validado de paciente. Por eso BDS, Ljung-Box, Diebold-Mariano y el MCS con bootstrap estacionario usan el orden de fila: se reportan porque la guía los exige, pero no son pruebas confirmatorias. Las sensibilidades principales siguen siendo las de la sección anterior.

**Supuestos de los residuos** (White con regresores [1, ŷ]; BDS m=2 sobre 3.000 residuos por su costo O(n²); Ljung-Box con 10 rezagos):

{T(dgs, ['familia','p_white','p_bds_m2','p_ljung_box_10','p_jarque_bera'], ['Familia','p White','p BDS','p Ljung-Box','p Jarque-Bera'])}
![ACF e histograma de residuos](acf_histograma.png)

**Friedman → Nemenyi** (mejor combinación por familia según RMSE externo medio, 5 folds): χ²={fr.statistic:.2f}, g.l.={k-1}, p={p(fr.pvalue)}, CD={cd:.2f}.

![Diagrama de diferencia crítica](cd_regresion.png)
"""
    if mcs_txt is not None:
        mt = mcs_txt.reset_index(); mt["p_mcs"] = mt["p_mcs"].map(p)
        md += f"\n**Model Confidence Set** (α=0,10; pérdida cuadrática; bootstrap estacionario, bloque 10, 1.000 réplicas, 15.000 pérdidas OOF, semilla 42):\n\n{T(mt, [mt.columns[0],'perdida_media','p_mcs','en_conjunto'], ['Modelo','Pérdida media','p MCS','¿En el conjunto?'], 2)}"
    md += f"""
**Comparaciones dirigidas: {b} frente al resto.** Incluye DM con corrección HLN, Wilcoxon sobre los folds, ajuste de Holm, IC 95 % bootstrap de ΔRMSE y d de Cohen. Clark-West no aplica porque las familias no están anidadas; Giacomini-White no aplica porque no hay ventana móvil.

{T(cs, ['modelo_b','delta_rmse','ic95_inf','ic95_sup','dm_hln','p_dm','p_dm_holm','p_wilcoxon','p_wilcoxon_holm','d_cohen_perdida','d_cohen_folds'], ['Frente a','ΔRMSE','IC inf','IC sup','DM-HLN','p DM','p DM Holm','p Wilcoxon','p Wilc. Holm','d (pérdida)','d (folds)'], 3)}
Con 5 folds, el p mínimo alcanzable con Wilcoxon es 0,0625. Con unas 44.000 observaciones, DM puede rechazar diferencias despreciables, así que la relevancia práctica se juzga con d: |d|<0,2 es despreciable.
"""
    (out / "complemento_guia.md").write_text(md)
    json.dump(dict(friedman=dict(chi2=float(fr.statistic), p=float(fr.pvalue)), nemenyi_cd=cd, rangos=rk.round(3).to_dict(), mejor=b,
                   mcs_incluidos=None if mcs_txt is None else mcs_txt.index[mcs_txt.en_conjunto].tolist()), open(out / "resumen.json", "w"), indent=2, ensure_ascii=False)
    print(md)

if __name__ == "__main__":
    main()
