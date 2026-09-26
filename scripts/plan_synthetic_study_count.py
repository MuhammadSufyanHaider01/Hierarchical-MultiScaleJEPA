#!/usr/bin/env python3
"""Lock the shared formal sample count from paired pilot anchor uncertainty."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import yaml
import h5py
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from synthetic_study.protocol import ANTENNA_LEVELS, TIME_PATCH_LEVELS

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def bootstrap_halfwidth(task, truth, pred, profile, seed, rounds=3000):
    rng=np.random.default_rng(seed)
    n=len(truth); vals=np.empty(rounds,dtype=np.float64)
    if task == "nrpos-outdoor":
        lo=np.asarray(profile["tasks"][task]["coordinate_min_m"])
        hi=np.asarray(profile["tasks"][task]["coordinate_max_m"])
        decoded=(pred+1.0)*0.5*(hi-lo)+lo
        decoded[:,hi==lo]=lo[hi==lo]
        errors=np.linalg.norm(decoded-truth,axis=1)
        for r in range(rounds):
            ix=rng.integers(0,n,n); vals[r]=errors[ix].mean()
    else:
        labels=np.unique(truth)
        groups=[np.flatnonzero(truth==label) for label in labels]
        for r in range(rounds):
            sampled=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
            guess=pred[sampled].argmax(axis=1)
            vals[r]=np.mean([np.mean(guess[truth[sampled]==label]==label) for label in labels])
    lo,hi=np.quantile(vals,[.025,.975])
    return {"half_width_95":float((hi-lo)/2),"ci95":[float(lo),float(hi)],"bootstrap_replicates":rounds,"sample_count":n}

def load_anchor(directory, method, axis, task, expected_level):
    base=directory/axis/task/method
    packet_path=base/"calibration_anchor.json"
    packet=json.loads(packet_path.read_text())
    if packet["stage"]!="full_information_calibration_anchor" or packet["method"]!=method or packet["axis"]!=axis or packet["task"]!=task:
        raise ValueError(f"anchor identity mismatch: {packet_path}")
    full=packet.get("condition", {})
    if int(full.get("level", -1)) != expected_level:
        raise ValueError(f"missing full-information condition {expected_level} in {packet_path}")
    expected_channels = 16 if axis == "antenna" else {"nrpos-outdoor": 4, "interference": 1, "jamming": 1}[task]
    expected_patches = 14 if axis == "antenna" else 56
    if axis == "antenna":
        actual_channels, actual_patches = expected_level, 14
    else:
        actual_channels, actual_patches = expected_channels, expected_level
    if actual_channels != expected_channels or actual_patches != expected_patches:
        raise ValueError(f"anchor does not match expected full-information control: {packet_path}")
    if not packet.get("gate", {}).get("gate_passed", False):
        raise ValueError(f"calibration gate did not pass: {packet_path}")
    pred_path=base/"calibration_anchor_predictions.npz"
    with np.load(pred_path) as archive:
        truth=archive["truth"].copy()
        pred=archive["layer_8"].copy()
    return {"packet":packet,"packet_path":packet_path,"prediction_path":pred_path,"truth":truth,"prediction":pred}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-run-root",type=Path,required=True,help="runs/.../pilot root (or formal root for formal calibration planning)")
    parser.add_argument("--profile",type=Path,default=ROOT/"configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--master",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=9025,help="bootstrap seed")
    parser.add_argument("--formal-seed",type=int,default=341902,help="locked realization seed for the formal master")
    args=parser.parse_args()
    if args.output.exists(): parser.error(f"refusing to overwrite locked count plan: {args.output}")
    profile=yaml.safe_load(args.profile.read_text())
    with h5py.File(args.master, "r") as master:
        if not bool(master.attrs.get("pilot", False)):
            raise ValueError("count planning requires the shared pilot master")
    targets={"nrpos-outdoor":float(profile["precision_targets"]["localization_error_95ci_half_width_m"]),"interference":float(profile["precision_targets"]["macro_accuracy_95ci_half_width"]),"jamming":float(profile["precision_targets"]["macro_accuracy_95ci_half_width"])}
    caps={key:int(value) for key,value in profile["precision_targets"]["source_count_caps"].items()}
    records={}; estimates=[]; passed_tasks=[]
    master_hash = sha(args.master)
    for task in ("nrpos-outdoor","interference","jamming"):
        target=targets[task]; task_records=[]; anchors=[]; missing=[]
        for axis,level in (("antenna",16),("time",56)):
            for method in ("p0","p2"):
                base=args.pilot_run_root/axis/task/method
                if not (base/"calibration_anchor.json").is_file() or not (base/"calibration_anchor_predictions.npz").is_file():
                    missing.append(str(base/"calibration_anchor.json")); continue
                anchor=load_anchor(args.pilot_run_root,method,axis,task,level)
                if anchor["packet"]["master_sha256"] != master_hash:
                    raise ValueError("pilot anchor does not use the supplied shared master")
                anchors.append(anchor)
        task_gates=[a["packet"]["gate"]["gate_passed"] for a in anchors]
        if missing or len(anchors)!=4 or not all(task_gates):
            records[task]={"status":"incomplete" if missing or len(anchors)!=4 else "blocked_by_failed_pilot_gate","pilot_gate_passed":False,"missing_anchors":missing,"observed_anchors":[{"method":a["packet"]["method"],"axis":a["packet"]["axis"],"status":a["packet"]["status"],"anchor":str(a["packet_path"])} for a in anchors],"cap":caps[task],"locked_task_count":None,"target_met_at_cap":False}
            continue
        reference_truth=None; reference_ids_hash=None; nvals=[]
        for anchor in anchors:
            packet=anchor["packet"]
            if reference_ids_hash is None:
                reference_ids_hash=packet["split_ids_sha256"]; reference_truth=anchor["truth"]
            elif packet["split_ids_sha256"]!=reference_ids_hash or not np.array_equal(anchor["truth"],reference_truth):
                raise ValueError("P0/P2 or antenna/time anchors are not paired on identical pilot IDs")
            ci=bootstrap_halfwidth(task,anchor["truth"],anchor["prediction"],profile,args.seed+len(estimates))
            required=int(math.ceil(ci["sample_count"]*(ci["half_width_95"]/target)**2))
            record={"task":task,"axis":packet["axis"],"method":packet["method"],"full_level":packet["condition"]["level"],"anchor_selection":str(anchor["packet_path"]),"anchor_sha256":sha(anchor["packet_path"]),"predictions_sha256":sha(anchor["prediction_path"]),"metric_ci":ci,"target_half_width":target,"estimated_count":required}
            task_records.append(record); nvals.append(required); estimates.append(required)
        required=max(nvals)
        task_count=min(max(required,max(x["metric_ci"]["sample_count"] for x in task_records)),caps[task])
        records[task]={"status":"passed","pilot_gate_passed":True,"anchors":task_records,"required_count_estimate":required,"cap":caps[task],"locked_task_count":task_count,"target_met_at_cap":required<=caps[task]}
        passed_tasks.append(task)
    if not passed_tasks:
        raise RuntimeError("no task passed all four paired pilot anchors; no formal count/master is authorized")
    # One immutable realization manifest is shared by every task that passed.
    common_required=max(records[task]["required_count_estimate"] for task in passed_tasks)
    common_cap=min(caps[task] for task in passed_tasks)
    total=min(max(common_required,512),common_cap)
    plan={"schema_version":1,"status":"locked","master":"one shared paired master for all tasks and availability sweeps","pilot_master":str(args.master.resolve()),"pilot_master_sha256":sha(args.master),"profile":str(args.profile.resolve()),"profile_sha256":sha(args.profile),"bootstrap_seed":args.seed,"formal_seed":args.formal_seed,"precision_method":"95% percentile bootstrap of the pilot selection split; classification resampled within each class","targets":targets,"per_task":records,"passed_tasks":passed_tasks,"shared_total_count":total,"total_count":total,"shared_cap":common_cap,"projected_shortfall":{task:(max(0,records[task]["required_count_estimate"]-total) if records[task]["pilot_gate_passed"] else None) for task in records},"note":"The shared count uses only tasks that passed all four pilot anchors and is capped by their strictest source-count cap. A failed task is skipped without blocking other tasks. Any precision miss is reported; the count is not extended using test data."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(plan,f,indent=2,sort_keys=True); f.write("\n")
    print(json.dumps({"output":str(args.output),"shared_total_count":total,"passed_tasks":passed_tasks,"required_by_task":{t:r.get("required_count_estimate") for t,r in records.items()},"projected_shortfall":plan["projected_shortfall"]},indent=2))
if __name__=="__main__": main()
