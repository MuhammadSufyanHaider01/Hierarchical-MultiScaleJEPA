#!/usr/bin/env python3
"""Audit a generated synthetic master and immutable split manifest before model runs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]

def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def require(condition:bool,message:str)->None:
    if not condition: raise ValueError(message)

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--profile",type=Path,default=ROOT/"configs/synthetic_trained_heads/factory_ofdma_v2.yml")
    parser.add_argument("--count-plan",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    master_path=args.master.resolve(); manifest_path=(args.manifest or master_path.parent/"split_manifest.json").resolve()
    record_path=master_path.parent/"generation_record.json"
    output=(args.output or master_path.parent/"master_audit.json").resolve()
    if output.exists(): parser.error(f"audit is immutable and already exists: {output}")
    profile=yaml.safe_load(args.profile.read_text()); plan=json.loads(args.count_plan.read_text())
    record=json.loads(record_path.read_text()); manifest=json.loads(manifest_path.read_text())
    profile_hash=sha256(args.profile); scene_path=ROOT/"src/synthetic_study/assets/factory_scene/scene0/scene.xml"
    scene_hash=sha256(scene_path); master_hash=sha256(master_path); manifest_hash=sha256(manifest_path)
    require(plan.get("status")=="locked" and plan.get("schema_version")==2,"expected the test-fraction-aware locked count plan")
    require(record.get("mode")=="formal" and record.get("count")==int(plan["shared_total_count"]),"generation record count/mode differs from locked plan")
    require(record.get("count_plan")==plan,"generation record embeds a different count plan")
    require(manifest.get("count_plan_sha256")==sha256(args.count_plan),"manifest is not bound to the supplied count plan")
    require(Path(manifest.get("count_plan", "")).resolve()==args.count_plan.resolve(),"manifest count-plan path mismatch")
    require(record.get("profile_sha256")==profile_hash==plan.get("profile_sha256"),"profile hash mismatch")
    require(record.get("scene_sha256")==scene_hash,"scene hash mismatch")
    require(record.get("data_sha256")==master_hash,"master SHA-256 mismatch")
    require(record.get("manifest_sha256")==manifest_hash,"manifest SHA-256 mismatch")
    require(manifest.get("master_sha256")==master_hash and manifest.get("profile_sha256")==profile_hash and manifest.get("scene_sha256")==scene_hash,"manifest provenance mismatch")
    require(not manifest.get("pilot",True) and "test" in manifest.get("realization_ids",{}),"formal test split is missing")
    split_ids=manifest["realization_ids"]
    split_names=("head_train","head_validation","calibration","exit_selection","test")
    require(set(split_ids)==set(split_names),f"unexpected formal split names: {sorted(split_ids)}")
    all_split_ids=[]
    for name in split_names:
        values=list(map(str,split_ids[name])); require(len(values)==len(set(values)),f"duplicate IDs in split {name}"); all_split_ids.extend(values)
    require(len(all_split_ids)==len(set(all_split_ids)),"formal realization splits overlap")
    require(len(all_split_ids)==int(plan["shared_total_count"]),"manifest total differs from locked count")
    require(manifest.get("counts")=={name:len(split_ids[name]) for name in split_names},"manifest split counts are incorrect")
    expected_fractions=profile["splits"]["fractions"]
    require(manifest.get("fractions")==expected_fractions,"manifest fractions differ from profile")
    expected_shape=(int(plan["shared_total_count"]),*map(int,profile["feature"]["master_shape"]))
    with h5py.File(master_path,"r") as h5:
        require(not bool(h5.attrs.get("pilot",True)),"master is marked as pilot")
        require(h5["features"].shape==expected_shape,f"feature shape mismatch: {h5['features'].shape} != {expected_shape}")
        require(h5["features"].dtype==np.dtype("float16"),f"unexpected feature dtype {h5['features'].dtype}")
        ids=list(map(str,h5["realization_id"].asstr()[:]))
        require(len(ids)==len(set(ids)),"master realization IDs are not unique")
        require(set(ids)==set(all_split_ids),"master IDs differ from split manifest")
        labels=np.asarray(h5["jammer_mode"][:],dtype=np.int64)
        binary=np.asarray(h5["interference_label"][:],dtype=np.int64)
        require(np.array_equal(binary,(labels>0).astype(np.int64)),"interference labels do not match jammer presence")
        label_counts=np.bincount(labels,minlength=6)
        jammed_count=len(ids)-len(ids)//2
        expected_labels=np.asarray([len(ids)//2]+[jammed_count//5 + int(i<jammed_count%5) for i in range(5)],dtype=np.int64)
        require(np.array_equal(label_counts,expected_labels),f"jammer label counts differ from balanced design: {label_counts.tolist()}")
        require(np.array_equal(np.bincount(binary,minlength=2),[len(ids)//2,len(ids)-len(ids)//2]),"interference label balance differs from design")
        coords=np.asarray(h5["receiver_xyz_m"][:],dtype=np.float32)
        require(coords.shape==(len(ids),3) and np.isfinite(coords).all(),"receiver coordinates are invalid")
        require(np.allclose(coords[:,2],float(profile["scenario"]["receiver"]["height_m"])),"receiver height differs from locked profile")
        min_xy=np.asarray(profile["scenario"]["receiver"]["coordinate_bounds_m"][:2],dtype=np.float32)
        require(np.all(coords[:,:2]>=min_xy[:,0]) and np.all(coords[:,:2]<=min_xy[:,1]),"receiver coordinates are outside the scene bounds")
        lo=float("inf"); hi=-float("inf"); finite_count=0
        for start in range(0,len(ids),16):
            block=np.asarray(h5["features"][start:min(start+16,len(ids))])
            require(np.isfinite(block).all(),f"non-finite feature values at sample block {start}")
            lo=min(lo,float(block.min())); hi=max(hi,float(block.max())); finite_count+=block.size
        require(finite_count==int(np.prod(expected_shape)),"not all feature values were audited")
    metadata_path=master_path.parent/"realization_metadata.jsonl"
    require(manifest.get("metadata_sha256")==sha256(metadata_path),"metadata SHA-256 mismatch")
    metadata=[]
    with metadata_path.open() as stream:
        for line in stream:
            if line.strip(): metadata.append(json.loads(line))
    require(len(metadata)==len(ids),"metadata row count differs from master")
    require([str(item["realization_id"]) for item in metadata]==ids,"metadata order/IDs differ from master")
    require(all(int(item["jammer_mode"])==int(label) for item,label in zip(metadata,labels,strict=True)),"metadata jammer labels differ from master")
    result={"schema_version":1,"status":"passed","mode":"formal","count":len(ids),"master":str(master_path),"master_sha256":master_hash,"manifest":str(manifest_path),"manifest_sha256":manifest_hash,"profile_sha256":profile_hash,"scene_sha256":scene_hash,"count_plan":str(args.count_plan.resolve()),"count_plan_sha256":sha256(args.count_plan),"split_counts":manifest["counts"],"split_ids_disjoint":True,"jammer_mode_counts":{str(i):int(label_counts[i]) for i in range(6)},"interference_counts":{"no_jammer":int(np.sum(binary==0)),"jammer_present":int(np.sum(binary==1))},"feature_shape":list(expected_shape),"feature_dtype":"float16","feature_min_db":lo,"feature_max_db":hi,"finite_feature_value_count":finite_count,"test_metrics_evaluated":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream: json.dump(result,stream,indent=2,sort_keys=True); stream.write("\n")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
