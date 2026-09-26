#!/usr/bin/env python3
"""Render paired validation-only synthetic availability packets."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
METHOD_DIR={"p0":"latentwave_ee","p2":"hierarchical"}
TASK_TITLE={"nrpos-outdoor":"NR Positioning (Factory-Scene Localization Proxy)","interference":"Interference (OFDMA Jammer-Presence Proxy)","jamming":"Jamming (OFDMA Jammer-Mode Proxy)"}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
def metric(task,truth,pred,profile):
    if task=="nrpos-outdoor":
        spec=profile["tasks"][task]; lo=np.asarray(spec["coordinate_min_m"]); hi=np.asarray(spec["coordinate_max_m"])
        decoded=(pred+1)*.5*(hi-lo)+lo; decoded[:,hi==lo]=lo[hi==lo]
        return np.linalg.norm(decoded-truth,axis=1).mean().item()
    guess=pred.argmax(1)
    return float(np.mean([np.mean(guess[truth==label]==label) for label in np.unique(truth)]))
def paired_bootstrap(task,truth,p0,p2,profile,seed=61025,rounds=3000):
    rng=np.random.default_rng(seed); n=len(truth); diff=np.empty(rounds)
    if task=="nrpos-outdoor":
        e0=np.linalg.norm((p0+1)*.5*(np.asarray(profile["tasks"][task]["coordinate_max_m"])-np.asarray(profile["tasks"][task]["coordinate_min_m"]))+np.asarray(profile["tasks"][task]["coordinate_min_m"])-truth,axis=1)
        e2=np.linalg.norm((p2+1)*.5*(np.asarray(profile["tasks"][task]["coordinate_max_m"])-np.asarray(profile["tasks"][task]["coordinate_min_m"]))+np.asarray(profile["tasks"][task]["coordinate_min_m"])-truth,axis=1)
        delta=e2-e0
        for b in range(rounds):diff[b]=delta[rng.integers(0,n,n)].mean()
        point=float(delta.mean())
    else:
        groups=[np.flatnonzero(truth==label) for label in np.unique(truth)]
        for b in range(rounds):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            score0=np.mean([np.mean(p0[ix][truth[ix]==c].argmax(1)==c) for c in np.unique(truth)])
            score2=np.mean([np.mean(p2[ix][truth[ix]==c].argmax(1)==c) for c in np.unique(truth)])
            diff[b]=score2-score0
        point=metric(task,truth,p2,profile)-metric(task,truth,p0,profile)
    lo,hi=np.quantile(diff,[.025,.975])
    return {"p2_minus_p0":float(point),"ci95":[float(lo),float(hi)],"bootstrap_replicates":rounds,"paired_n":n}
def load_packet(run_root,mode,axis,task,method,include_test=False):
    base=run_root/mode/axis/task/method
    selection_path=base/"selection.json"
    if include_test:
        if mode!="formal":raise ValueError("held-out test reports require formal mode")
        selection=json.loads(selection_path.read_text())
        packet_path=base/"test_metrics.json"; packet=json.loads(packet_path.read_text())
        if packet.get("status")!="test_complete" or packet.get("stage")!="test":raise ValueError(f"not a completed held-out test packet: {packet_path}")
        if packet.get("selection_artifact_sha256")!=sha(selection_path):raise ValueError(f"test packet is not bound to the current reviewed selection: {packet_path}")
        approval_path=base/"review_approval.json"
        approval=json.loads(approval_path.read_text())
        if approval.get("approved") is not True or not approval.get("reviewer") or approval.get("selection_sha256")!=sha(selection_path):raise ValueError(f"review approval does not authorize this exact selection: {approval_path}")
        if packet.get("review_approval_sha256")!=sha(approval_path):raise ValueError(f"test packet is not bound to the current approval: {packet_path}")
        preds_path=base/"test_predictions.npz"
        if packet.get("test_predictions_sha256")!=sha(preds_path):raise ValueError(f"test prediction hash mismatch: {preds_path}")
    else:
        packet_path=selection_path; packet=json.loads(packet_path.read_text())
        if packet.get("status")!="selection_complete_review_required" or packet.get("stage")!="select":raise ValueError(f"not a finalized validation selection artifact: {packet_path}")
        preds_path=base/"selection_predictions.npz"
    return packet,packet_path,preds_path,base
def plot_rows(rows,task,axis,metric_name,higher,title,path,methods):
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    styles={"p0":("#2563eb","o"),"p2":("#dc2626","s")}
    for method in methods:
        selected=[r for r in rows if r["method"]==method]
        color,marker=styles[method]
        xs=[r["level"] for r in selected]; ys=[r["selected_metric"] for r in selected]
        label="LatentWave + EE" if method=="p0" else "Proposed hierarchical approach"
        ax.plot(xs,ys,marker=marker,color=color,linewidth=2,label=label)
        for row in selected:
            text=f"L{row['selected_exit']}"
            ax.annotate(text,(row["level"],row["selected_metric"]),xytext=(0,9 if method=="p0" else -15),textcoords="offset points",ha="center",fontsize=8,color=color)
    ax.set_title(title); ax.set_xlabel("Antenna channels C" if axis=="antenna" else "Temporal patches Tₚ")
    ax.set_ylabel("Mean distance error (m)" if task=="nrpos-outdoor" else "Macro accuracy")
    ax.grid(True,alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=tuple(TASK_TITLE),required=True)
    parser.add_argument("--axis",choices=("antenna","time"),required=True)
    parser.add_argument("--mode",choices=("pilot","formal"),default="formal")
    parser.add_argument("--profile",type=Path,default=ROOT/"configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--run-root",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--bootstrap-seed",type=int,default=61025)
    parser.add_argument("--include-test",action="store_true",help="render a separate report from already-approved formal test packets")
    args=parser.parse_args()
    profile=yaml.safe_load(args.profile.read_text())
    run_root=(ROOT/profile["outputs"]["run_root"]).resolve() if args.run_root is None else args.run_root.resolve()
    exp_root=(ROOT/profile["outputs"]["experiment_root"]).resolve() if args.output_root is None else args.output_root.resolve()
    axis_name="vary_antenna" if args.axis=="antenna" else "vary_time_samples"
    task_root=exp_root/axis_name/args.task
    report_root=task_root/"test" if args.include_test else task_root
    if report_root.exists() and any(report_root.iterdir()):raise FileExistsError(f"report output exists; refusing to overwrite: {report_root}")
    report_root.mkdir(parents=True,exist_ok=True)
    packets={}; arrays={}; rows=[]
    for method in ("p0","p2"):
        packet,path,pred_path,run_dir=load_packet(run_root,args.mode,args.axis,args.task,method,args.include_test)
        packets[method]=(packet,path,pred_path,run_dir)
        with np.load(pred_path) as archive:
            for level,condition in sorted(packet["conditions"].items(), key=lambda item: int(item[0])):
                truth=archive[f"level_{level}_truth"].copy()
                selected=int(condition["selected_exit"])
                selected_prediction=archive[f"level_{level}_layer_{selected}"].copy()
                arrays[(method,int(level))]=(truth,selected_prediction)
                row={"method":method,"method_label":"LatentWave + EE" if method=="p0" else "Proposed hierarchical approach","task":args.task,"axis":args.axis,"level":int(level),"condition":condition["condition"],"channels":condition["condition"]["channels"],"array_element_indices":condition["condition"]["array_element_indices"],"temporal_patches":condition["condition"]["temporal_patches"],"time_columns":condition["condition"]["time_columns"],"token_count":condition["condition"]["token_count"],"selected_exit":selected,"selected_metric":condition["selected_metric"],"l8_metric":condition["fixed_exit_metrics"]["8"],"l8_gap":condition["l8_gap"],"l8_relative_gap":condition["l8_relative_gap"],"flops_approx":condition["selected_exit_flops_approx"],"latency_ms_batch1":condition["selected_exit_latency_ms_batch1"],"fixed_exit_metrics":condition["fixed_exit_metrics"],"source_checkpoint":packet["source_checkpoint"],"source_checkpoint_sha256":packet["source_checkpoint_sha256"],"master_sha256":packet["master_sha256"],"split_ids_sha256":packet["split_ids_sha256"]}
                rows.append(row)
    a,b=packets["p0"][0],packets["p2"][0]
    if (a["task"],a["axis"],a["metric"],a["master_sha256"],a["split_ids_sha256"])!=(b["task"],b["axis"],b["metric"],b["master_sha256"],b["split_ids_sha256"]):raise ValueError("P0/P2 packets are not paired on the same task, axis, master, and selection IDs")
    if sorted(a["conditions"],key=int)!=sorted(b["conditions"],key=int):raise ValueError("P0/P2 condition levels differ")
    paired=[]
    for level in sorted(a["conditions"],key=int):
        truth0,p0=arrays[("p0",int(level))]; truth2,p2=arrays[("p2",int(level))]
        if not np.array_equal(truth0,truth2):raise ValueError(f"labels differ at condition {level}")
        interval=paired_bootstrap(args.task,truth0,p0,p2,profile,args.bootstrap_seed+int(level))
        paired.append({"level":int(level),**interval,"p0_selected_exit":a["conditions"][level]["selected_exit"],"p2_selected_exit":b["conditions"][level]["selected_exit"],"p0_metric":a["conditions"][level]["selected_metric"],"p2_metric":b["conditions"][level]["selected_metric"]})
    for method in ("p0","p2"):
        md=report_root/METHOD_DIR[method]; md.mkdir(parents=True,exist_ok=True)
        subrows=[r for r in rows if r["method"]==method]
        csv_path=md/"metrics.csv"
        columns=["method","task","axis","level","channels","array_element_indices","temporal_patches","time_columns","token_count","selected_exit","selected_metric","l8_metric","l8_gap","l8_relative_gap","flops_approx","latency_ms_batch1","source_checkpoint","source_checkpoint_sha256","master_sha256","split_ids_sha256"]
        with csv_path.open("x",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows([{k:(json.dumps(r[k]) if isinstance(r[k],(list,dict)) else r[k]) for k in columns} for r in subrows])
        (md/"metrics.json").write_text(json.dumps({"selection":packets[method][0],"rows":subrows},indent=2,sort_keys=True)+"\n")
        (md/"README.md").write_text(f"# {TASK_TITLE[args.task]} — {packets[method][0]['method']}\n\nValidation-only synthetic-domain availability results. Synthetic tasks are proxies, not the measured LatentWave downstream datasets.\n\nSource selection packet: `{packets[method][1]}` (SHA-256 `{sha(packets[method][1])}`).\nMaster SHA-256: `{a['master_sha256']}`. Split-ID SHA-256: `{a['split_ids_sha256']}`.\n")
        table="| Level | Selected exit | Selection metric | L8 metric | L8 gap | FLOPs (approx.) | Batch-1 latency (ms) |\n|---:|---:|---:|---:|---:|---:|---:|\n"
        for r in subrows:table+=f"| {r['level']} | L{r['selected_exit']} | {r['selected_metric']:.5g} | {r['l8_metric']:.5g} | {r['l8_gap']:.5g} | {r['flops_approx']} | {r['latency_ms_batch1']:.5g} |\n"
        (md/"metrics.md").write_text(f"# {TASK_TITLE[args.task]} — {r['method_label']}\n\n{table}")
        plot_rows(rows,args.task,args.axis,packets[method][0]["metric"],packets[method][0]["higher_is_better"],TASK_TITLE[args.task]+" — " + ("LatentWave + EE" if method=="p0" else "Proposed hierarchical approach"),md/"performance.png",[method])
        # This PDF is the same individual method figure as the PNG.
        plot_rows(rows,args.task,args.axis,packets[method][0]["metric"],packets[method][0]["higher_is_better"],TASK_TITLE[args.task]+" — " + ("LatentWave + EE" if method=="p0" else "Proposed hierarchical approach"),md/"performance.pdf",[method])
    with (report_root/"paired_metrics.csv").open("x",newline="") as f:
        fields=["level","p0_selected_exit","p2_selected_exit","p0_metric","p2_metric","p2_minus_p0","ci95_lower","ci95_upper","paired_n"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in paired:w.writerow({"level":r["level"],"p0_selected_exit":r["p0_selected_exit"],"p2_selected_exit":r["p2_selected_exit"],"p0_metric":r["p0_metric"],"p2_metric":r["p2_metric"],"p2_minus_p0":r["p2_minus_p0"],"ci95_lower":r["ci95"][0],"ci95_upper":r["ci95"][1],"paired_n":r["paired_n"]})
    comparison_kind="held-out test" if args.include_test else "validation selection"
    (report_root/"paired_metrics.md").write_text(f"# Paired {comparison_kind} comparison\n\nP2 minus P0 paired bootstrap interval; for localization, negative favors P2; for accuracy, positive favors P2.\n\n| Level | P0 exit / metric | P2 exit / metric | P2 − P0 | 95% paired CI | n |\n|---:|---|---|---:|---|---:|\n"+"".join(f"| {r['level']} | L{r['p0_selected_exit']} / {r['p0_metric']:.5g} | L{r['p2_selected_exit']} / {r['p2_metric']:.5g} | {r['p2_minus_p0']:.5g} | [{r['ci95'][0]:.5g}, {r['ci95'][1]:.5g}] | {r['paired_n']} |\n" for r in paired))
    (report_root/"paired_metrics.json").write_text(json.dumps({"task":args.task,"axis":args.axis,"stage":comparison_kind,"metric":a["metric"],"synthetic_proxy":True,"p0_packet_sha256":sha(packets["p0"][1]),"p2_packet_sha256":sha(packets["p2"][1]),"rows":paired},indent=2,sort_keys=True)+"\n")
    title=TASK_TITLE[args.task]+(" — held-out test" if args.include_test else " — validation selection")
    plot_rows(rows,args.task,args.axis,a["metric"],a["higher_is_better"],title,report_root/"comparison.png",["p0","p2"])
    plot_rows(rows,args.task,args.axis,a["metric"],a["higher_is_better"],title,report_root/"comparison.pdf",["p0","p2"])
    report_label="approved held-out test" if args.include_test else "validation-only; held-out test metrics are not included"
    (report_root/"README.md").write_text(f"# {title} — paired synthetic availability study\n\nThis is an {report_label} report. Task definition: `{profile['tasks'][args.task]['synthetic_name']}`. The run is supplementary synthetic-domain evidence and does not recreate measured LatentWave data.\n\nP0 source: `{packets['p0'][1]}` (SHA-256 `{sha(packets['p0'][1])}`).\nP2 source: `{packets['p2'][1]}` (SHA-256 `{sha(packets['p2'][1])}`).\nMaster: `{a['master_sha256']}`; evaluation IDs: `{a['split_ids_sha256']}`.\n")
    exp_root.mkdir(parents=True, exist_ok=True)
    matrix = exp_root / "README.md"
    if not matrix.exists():
        matrix.write_text("# Synthetic trained-head availability study\n\nSupplementary results in an OFDMA factory-scenario proxy domain; these do not reproduce the measured LatentWave tasks, and the measured-data experiment remains primary evidence. Each report is generated only from validation selection artifacts until a separate review approval is recorded.\n\n| Family | Task proxy | Status |\n|---|---|---|\n| Antenna channels | NR positioning / factory localization | Reported per task when selection artifacts are reviewed |\n| Antenna channels | Interference / jammer presence | Reported per task when selection artifacts are reviewed |\n| Antenna channels | Jamming / jammer mode | Reported per task when selection artifacts are reviewed |\n| Temporal patches | NR positioning / factory localization | Reported per task when selection artifacts are reviewed |\n| Temporal patches | Interference / jammer presence | Reported per task when selection artifacts are reviewed |\n| Temporal patches | Jamming / jammer mode | Reported per task when selection artifacts are reviewed |\n\nProfiles, generator code, and immutable master/manifests are versioned under `configs/synthetic_trained_heads/`, `src/synthetic_study/`, and `data/synthetic_trained_heads/` (generated data are ignored by Git).\n")
    print(json.dumps({"output":str(report_root),"conditions":len(paired),"status":"test_reported" if args.include_test else "validation_reported_test_not_run"},indent=2))
if __name__=="__main__":main()
