#!/usr/bin/env python3
"""Render paired reports for full-information versus condition-specific heads."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from synthetic_trained_head_experiment import condition_spec, primary_metric
from synthetic_trained_head_experiment import decode_positions

METHOD_DIR={"p0":"latentwave_ee","p2":"hierarchical"}
TASK_TITLE={"nrpos-outdoor":"NR Positioning (Factory-Scene Localization Proxy)","interference":"Interference (OFDMA Jammer-Presence Proxy)","jamming":"Jamming (OFDMA Jammer-Mode Proxy)"}

def sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def paired_delta(task:str,truth:np.ndarray,full:np.ndarray,condition:np.ndarray,profile:dict,seed:int,rounds:int=3000)->dict:
    rng=np.random.default_rng(seed); n=len(truth); values=np.empty(rounds)
    if task=="nrpos-outdoor":
        ef=np.linalg.norm(decode_positions(full,profile)-truth,axis=1)
        ec=np.linalg.norm(decode_positions(condition,profile)-truth,axis=1)
        delta=ec-ef
        for i in range(rounds): values[i]=delta[rng.integers(0,n,n)].mean()
        point=float(delta.mean())
    else:
        labels=np.unique(truth); groups=[np.flatnonzero(truth==label) for label in labels]
        for i in range(rounds):
            sample=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
            values[i]=primary_metric(task,truth[sample],condition[sample],profile)-primary_metric(task,truth[sample],full[sample],profile)
        point=primary_metric(task,truth,condition,profile)-primary_metric(task,truth,full,profile)
    lower,upper=np.quantile(values,[.025,.975])
    return {"condition_specific_minus_full_information":point,"ci95":[float(lower),float(upper)],"bootstrap_replicates":rounds,"paired_n":n}

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=tuple(TASK_TITLE),required=True)
    parser.add_argument("--axis",choices=("antenna","time"),required=True)
    parser.add_argument("--profile",type=Path,default=ROOT/"configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--bootstrap-seed",type=int,default=71031)
    args=parser.parse_args()
    profile=yaml.safe_load(args.profile.read_text())
    run=(ROOT/profile["outputs"]["run_root"]).resolve() if args.run_root is None else args.run_root.resolve()
    exp=(ROOT/profile["outputs"]["experiment_root"]).resolve() if args.output_root is None else args.output_root.resolve()
    axis_dir="vary_antenna" if args.axis=="antenna" else "vary_time_samples"
    # Keep condition-specific synthetic diagnostics in their own experiment family,
    # parallel to original_data and synthetic_data, rather than nested in the latter.
    root=exp.parent/"condition_specific"/axis_dir/args.task
    if root.exists() and any(root.iterdir()): raise FileExistsError(f"diagnostic report is immutable and already exists: {root}")
    levels=condition_spec(args.task,args.axis)[0]
    packets={}; archives={}; rows=[]
    for method in ("p0","p2"):
        full_dir=run/"formal"/args.axis/args.task/method
        full_path=full_dir/"selection.json"; full=json.loads(full_path.read_text())
        full_pred=full_dir/"selection_predictions.npz"
        if full.get("status")!="selection_complete_review_required": raise ValueError(f"primary selection is not finalized: {full_path}")
        packets[(method,"full")]=(full,full_path,full_pred)
        for level in levels:
            diag_dir=run/"condition_specific"/"formal"/args.axis/args.task/method/f"level_{level}"
            diag_path=diag_dir/"selection.json"; diag=json.loads(diag_path.read_text()); diag_pred=diag_dir/"selection_predictions.npz"
            if diag.get("status")!="selection_complete_review_required" or diag.get("stage")!="select": raise ValueError(f"condition-specific selection is not finalized: {diag_path}")
            if diag.get("master_sha256")!=full.get("master_sha256") or diag.get("split_ids_sha256")!=full.get("split_ids_sha256") or diag.get("profile_sha256")!=full.get("profile_sha256"):
                raise ValueError(f"condition-specific artifact is not paired to the primary split: {diag_path}")
            train=json.loads(Path(diag["training_record"]).read_text())
            if train.get("condition_specific") is not True or train.get("training_condition",{}).get("level")!=level:
                raise ValueError(f"condition-specific head provenance mismatch: {diag["training_record"]}")
            packets[(method,level)]=(diag,diag_path,diag_pred)
    for level in levels:
        for method in ("p0","p2"):
            full,full_path,full_npz=packets[(method,"full")]
            diag,diag_path,diag_npz=packets[(method,level)]
            full_exit=int(full["conditions"][str(level)]["selected_exit"])
            diag_exit=int(diag["conditions"][str(level)]["selected_exit"])
            with np.load(full_npz) as f, np.load(diag_npz) as d:
                truth=f[f"level_{level}_truth"].copy(); diag_truth=d[f"level_{level}_truth"].copy()
                full_pred=f[f"level_{level}_layer_{full_exit}"].copy()
                diag_pred=d[f"level_{level}_layer_{diag_exit}"].copy()
            if not np.array_equal(truth,diag_truth): raise ValueError(f"paired labels mismatch at {args.task}/{args.axis}={level}/{method}")
            interval=paired_delta(args.task,truth,full_pred,diag_pred,profile,args.bootstrap_seed+level+(0 if method=="p0" else 10000))
            fc=full["conditions"][str(level)]; dc=diag["conditions"][str(level)]
            rows.append({"task":args.task,"axis":args.axis,"level":level,"method":method,"method_label":"LatentWave + EE" if method=="p0" else "Proposed hierarchical approach","full_information_selected_exit":full_exit,"condition_specific_selected_exit":diag_exit,"full_information_metric":fc["selected_metric"],"condition_specific_metric":dc["selected_metric"],**interval,"full_information_l8_metric":fc["fixed_exit_metrics"]["8"],"condition_specific_l8_metric":dc["fixed_exit_metrics"]["8"],"full_information_flops_approx":fc["selected_exit_flops_approx"],"full_information_latency_ms_batch1":fc["selected_exit_latency_ms_batch1"],"condition_specific_flops_approx":dc["selected_exit_flops_approx"],"condition_specific_latency_ms_batch1":dc["selected_exit_latency_ms_batch1"],"master_sha256":diag["master_sha256"],"split_ids_sha256":diag["split_ids_sha256"],"full_information_selection":str(full_path),"condition_specific_selection":str(diag_path),"full_information_selection_sha256":sha(full_path),"condition_specific_selection_sha256":sha(diag_path)})
    root.mkdir(parents=True,exist_ok=True)
    colors={"p0":"#2563eb","p2":"#dc2626"}
    for method in ("p0","p2"):
        sub=[r for r in rows if r["method"]==method]; out=root/METHOD_DIR[method]; out.mkdir()
        with (out/"metrics.csv").open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(sub[0])); writer.writeheader(); writer.writerows(sub)
        (out/"metrics.json").write_text(json.dumps(sub,indent=2,sort_keys=True)+"\n")
        table="| Level | Full-info exit / metric | Condition-specific exit / metric | Δ (95% paired CI) |\n|---:|---|---|---|\n"
        for r in sub: table+=f"| {r['level']} | L{r['full_information_selected_exit']} / {r['full_information_metric']:.5g} | L{r['condition_specific_selected_exit']} / {r['condition_specific_metric']:.5g} | {r['condition_specific_minus_full_information']:.5g} [{r['ci95'][0]:.5g}, {r['ci95'][1]:.5g}] |\n"
        (out/"metrics.md").write_text(f"# {TASK_TITLE[args.task]} — {sub[0]['method_label']}\n\nValidation-only, paired condition-specific-head diagnostic.\n\n"+table)
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        ax.plot(levels,[r["full_information_metric"] for r in sub],marker="o",color=colors[method],linestyle="--",label="Full-information-trained head")
        ax.plot(levels,[r["condition_specific_metric"] for r in sub],marker="s",color=colors[method],label="Condition-specific head")
        ax.set_title(TASK_TITLE[args.task]+" — "+sub[0]["method_label"]+" (condition-specific diagnostic)")
        ax.set_xlabel("Antenna channels C" if args.axis=="antenna" else "Temporal patches Tₚ")
        ax.set_ylabel("Mean distance error (m)" if args.task=="nrpos-outdoor" else "Macro accuracy")
        ax.grid(True,alpha=.25); ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(out/"performance.png",dpi=220); fig.savefig(out/"performance.pdf"); plt.close(fig)
        (out/"README.md").write_text(f"# {TASK_TITLE[args.task]} — {sub[0]['method_label']} condition-specific diagnostic\n\nEach condition has its own task heads trained on the same formal head-train IDs; normalization is inherited from that method/sweep’s full-information fit. Metrics and exit choice use only the separate exit-selection split.\n")
    with (root/"paired_diagnostic.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root/"paired_diagnostic.json").write_text(json.dumps({"task":args.task,"axis":args.axis,"rows":rows},indent=2,sort_keys=True)+"\n")
    md="| Level | Method | Full-info exit / metric | Condition-specific exit / metric | Δ (95% paired CI) |\n|---:|---|---|---|---|\n"
    for row in rows: md+=f"| {row['level']} | {row['method_label']} | L{row['full_information_selected_exit']} / {row['full_information_metric']:.5g} | L{row['condition_specific_selected_exit']} / {row['condition_specific_metric']:.5g} | {row['condition_specific_minus_full_information']:.5g} [{row['ci95'][0]:.5g}, {row['ci95'][1]:.5g}] |\n"
    (root/"paired_diagnostic.md").write_text(f"# {TASK_TITLE[args.task]} — paired condition-specific diagnostic\n\n"+md)
    fig,ax=plt.subplots(figsize=(8,5))
    for method in ("p0","p2"):
        sub=[r for r in rows if r["method"]==method]; color=colors[method]
        label="LatentWave + EE" if method=="p0" else "Proposed hierarchical approach"
        ax.plot(levels,[r["full_information_metric"] for r in sub],marker="o",color=color,linestyle="--",label=label+" — full-info heads")
        ax.plot(levels,[r["condition_specific_metric"] for r in sub],marker="s",color=color,label=label+" — condition-specific heads")
    ax.set_title(TASK_TITLE[args.task]+" — condition-specific diagnostic")
    ax.set_xlabel("Antenna channels C" if args.axis=="antenna" else "Temporal patches Tₚ")
    ax.set_ylabel("Mean distance error (m)" if args.task=="nrpos-outdoor" else "Macro accuracy")
    ax.grid(True,alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(root/"comparison.png",dpi=220); fig.savefig(root/"comparison.pdf"); plt.close(fig)
    (root/"README.md").write_text(f"# {TASK_TITLE[args.task]} — condition-specific-head diagnostic\n\nThis validation-only diagnostic asks whether the primary trend depends on training heads only at each sweep’s full-information condition. It uses paired realizations and shared split IDs; it is synthetic-domain evidence, not a recreation of measured LatentWave data.\n")
    print(json.dumps({"output":str(root),"rows":len(rows),"status":"validation_only"},indent=2))

if __name__=="__main__": main()
