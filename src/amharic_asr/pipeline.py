"""Reproducible data and benchmark utilities."""
from __future__ import annotations
import argparse, csv, hashlib, json, random, re, shutil, zipfile, wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
ETHIOPIC = re.compile(r"[\u1200-\u137f]")

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while b:=f.read(chunk): h.update(b)
    return h.hexdigest()

def read_jsonl(path):
    with open(path,encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]

def write_jsonl(rows,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def wav_info(path):
    try:
        with wave.open(str(path),"rb") as w:
            return {"sample_rate":w.getframerate(),"channels":w.getnchannels(),"sample_width":w.getsampwidth(),"frames":w.getnframes(),"duration":w.getnframes()/w.getframerate()}
    except Exception as e: return {"error":str(e)}

def iter_parquet(source):
    import pandas as pd
    paths=sorted(Path(source).rglob("*.parquet")) if Path(source).is_dir() else [Path(source)]
    for p in paths:
        for i,row in pd.read_parquet(p).iterrows():
            audio=row.get("audio",{}) or {}
            if isinstance(audio,str):
                try: audio=json.loads(audio)
                except Exception: audio={"path":audio}
            yield {"source_file":str(p),"source_row_id":f"{p.name}:{i}","text":str(row.get("text","")),"dialect":str(row.get("dialect","")),"speaker_id":str(row.get("speaker_id","")),"gender":str(row.get("gender","")),"audio_path":audio.get("path","")}

def audit(source,output,expected_sha256=None):
    src,out=Path(source),Path(output); out.mkdir(parents=True,exist_ok=True)
    archive_hash=sha256_file(src) if src.is_file() else None
    if expected_sha256 and archive_hash and archive_hash!=expected_sha256: raise SystemExit(f"SHA-256 mismatch: expected {expected_sha256}, got {archive_hash}")
    root=src if src.is_dir() else out/"extracted"
    if src.is_file() and not root.exists():
        with zipfile.ZipFile(src) as z: z.extractall(root)
    rows=list(iter_parquet(root)); texts=Counter(" ".join(r["text"].split()) for r in rows); speakers=defaultdict(set); counts=Counter(); flags=[]
    for r in rows:
        d=r["dialect"].lower().replace(" ","_"); counts[d]+=1; speakers[d].add(r["speaker_id"]); t=r["text"]; f=[]
        if "�" in t:f.append("replacement_character")
        if not ETHIOPIC.search(t):f.append("no_ethiopic")
        if len(t)==500:f.append("exact_500_chars")
        if f:flags.append({"source_row_id":r["source_row_id"],"flags":f})
    report={"archive_sha256":archive_hash,"rows":len(rows),"dialect_counts":counts,"speakers_by_dialect":{k:len(v) for k,v in speakers.items()},"unique_normalized_transcripts":len(texts),"repeated_transcript_rows":sum(n for n in texts.values() if n>1),"flags":len(flags),"source":str(source)}
    json.dump(report,open(out/"report.json","w",encoding="utf-8"),ensure_ascii=False,indent=2,default=int)
    write_jsonl(rows,out/"rows.jsonl"); json.dump(flags,open(out/"issues.json","w",encoding="utf-8"),ensure_ascii=False,indent=2); return report

def review_sample(audit_path,output,n_per_dialect,seed):
    rows=read_jsonl(audit_path); rng=random.Random(seed); selected=[]
    for d in sorted({r.get("dialect","") for r in rows}): selected += rng.sample([r for r in rows if r.get("dialect","")==d],min(n_per_dialect,sum(r.get("dialect","")==d for r in rows)))
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(r.keys() for r in selected)))+["transcript_decision","dialect_decision","corrected_transcript","corrected_dialect","reviewer_id","notes"]
    with open(output,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=dict.fromkeys(fields)); w.writeheader(); w.writerows(selected)

def resample_wav(source, target, sample_rate):
    import librosa, soundfile as sf
    audio, original_rate = librosa.load(str(source), sr=None, mono=True)
    if original_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=original_rate, target_sr=sample_rate)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), audio, sample_rate, subtype="PCM_16")

def prepare(source,audit_path,reviews,output,max_seconds,sample_rate=16000,allow_unreviewed=False):
    rows=read_jsonl(audit_path); decisions={}
    if reviews and Path(reviews).exists():
        with open(reviews,encoding="utf-8") as f:
            for r in csv.DictReader(f): decisions[r.get("source_row_id","")]=r
    if not allow_unreviewed and not decisions: raise SystemExit("No review decisions found; use --allow-unreviewed only for a non-release smoke run")
    out=Path(output); manifest=[]
    for r in rows:
        dec=decisions.get(r["source_row_id"],{})
        if dec.get("transcript_decision")=="exclude" or dec.get("dialect_decision")=="exclude": continue
        audio=Path(source)/r["audio_path"] if r["audio_path"] else None
        if not audio or not audio.exists(): continue
        info=wav_info(audio)
        if "error" in info or info["duration"]>max_seconds: continue
        target=out/"audio"/(r["source_row_id"].replace("/","_").replace(":","_")+".wav"); resample_wav(audio,target,sample_rate)
        duration=wav_info(target)["duration"]
        manifest.append({"id":r["source_row_id"],"audio":str(target),"text":dec.get("corrected_transcript") or r["text"],"dialect":dec.get("corrected_dialect") or r["dialect"],"speaker_id":r["speaker_id"],"gender":r["gender"],"duration":duration,"source_row_id":r["source_row_id"]})
    write_jsonl(manifest,out/"manifest.jsonl"); json.dump({"rows":len(manifest),"sample_rate":sample_rate,"max_seconds":max_seconds},open(out/"provenance.json","w",encoding="utf-8"),indent=2)

def make_splits(manifest_path,output,speaker_scope,seed,test_fraction,validation_fraction):
    rows=read_jsonl(manifest_path); rng=random.Random(seed)
    for r in rows:r["split_speaker"]=r["speaker_id"] if speaker_scope=="global" else f"{r['dialect']}::{r['speaker_id']}"
    speakers=sorted({r["split_speaker"] for r in rows}); rng.shuffle(speakers); nt=max(1,round(len(speakers)*test_fraction)); nv=max(1,round(len(speakers)*validation_fraction)); test_s=set(speakers[:nt]); val_s=set(speakers[nt:nt+nv]); splits={"test":[],"validation":[],"train":[]}
    for r in rows:splits["test" if r["split_speaker"] in test_s else "validation" if r["split_speaker"] in val_s else "train"].append(r)
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items():write_jsonl(v,out/f"{k}.jsonl")
    sp=lambda d:{r["split_speaker"] for r in d}; prompt=lambda d:{" ".join(r["text"].split()) for r in d}
    report={"counts":{k:len(v) for k,v in splits.items()},"speaker_overlap":{f"{a}_{b}":len(sp(splits[a])&sp(splits[b])) for a,b in (("train","validation"),("train","test"),("validation","test"))},"prompt_overlap_test_train":len(prompt(splits["test"])&prompt(splits["train"])),"speaker_scope":speaker_scope,"seed":seed}
    json.dump(report,open(out/"leakage_report.json","w",encoding="utf-8"),indent=2)

def main_audit():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",required=True); p.add_argument("--expected-sha256"); a=p.parse_args(); audit(a.source,a.output,a.expected_sha256)
def main_review():
    p=argparse.ArgumentParser(); p.add_argument("--audit",required=True); p.add_argument("--output",required=True); p.add_argument("--n-per-dialect",type=int,default=30); p.add_argument("--seed",type=int,default=42); a=p.parse_args(); review_sample(a.audit,a.output,a.n_per_dialect,a.seed)
def main_prepare():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--audit",required=True); p.add_argument("--reviews",required=True); p.add_argument("--output",required=True); p.add_argument("--sample-rate",type=int,default=16000); p.add_argument("--max-seconds",type=float,default=30); p.add_argument("--allow-unreviewed",action="store_true"); a=p.parse_args(); prepare(a.source,a.audit,a.reviews,a.output,a.max_seconds,a.sample_rate,a.allow_unreviewed)
def main_splits():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",required=True); p.add_argument("--output",required=True); p.add_argument("--speaker-scope",default="dialect_local",choices=["dialect_local","global"]); p.add_argument("--seed",type=int,default=42); p.add_argument("--test-speakers",type=float,default=.2); p.add_argument("--validation-speakers",type=float,default=.1); a=p.parse_args(); make_splits(a.manifest,a.output,a.speaker_scope,a.seed,a.test_speakers,a.validation_speakers)
