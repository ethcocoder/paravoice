"""Whisper evaluation and fine-tuning helpers; imports heavy ML libraries only at runtime."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path

def load_rows(path):
    with open(path,encoding='utf-8') as f:return [json.loads(x) for x in f if x.strip()]

def score(model_name, manifest, output, batch_size=2, max_steps=None, train=False, max_items=None, fp16=False, grad_ckpt=False):
    import torch
    from datasets import Dataset, Audio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    rows=load_rows(manifest); rows=rows[:max_items] if max_items else rows
    ds=Dataset.from_list(rows).cast_column('audio',Audio(sampling_rate=16000))
    processor=WhisperProcessor.from_pretrained(model_name,language='am',task='transcribe')
    model=WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language='am'; model.generation_config.task='transcribe'; model.generation_config.forced_decoder_ids=None
    def prep(batch):
        audio=batch['audio']; batch['input_features']=processor.feature_extractor(audio['array'],sampling_rate=audio['sampling_rate']).input_features[0]; batch['labels']=processor.tokenizer(batch['text']).input_ids; return batch
    ds=ds.map(prep,remove_columns=ds.column_names)
    def collate(features):
        import torch
        x=torch.tensor([f['input_features'] for f in features]); labels=[f['labels'] for f in features]; labels=processor.tokenizer.pad({'input_ids':labels},padding=True,return_tensors='pt').input_ids; labels[labels==processor.tokenizer.pad_token_id]=-100; return {'input_features':x,'labels':labels}
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    if train:
        args=Seq2SeqTrainingArguments(output_dir=str(out),per_device_train_batch_size=batch_size,gradient_accumulation_steps=8,learning_rate=1e-6,warmup_steps=100,max_steps=max_steps or 1000,fp16=fp16 and torch.cuda.is_available(),gradient_checkpointing=grad_ckpt,logging_steps=10,save_steps=100,eval_strategy='no',report_to='none',remove_unused_columns=False)
        trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=ds,data_collator=collate,processing_class=processor); trainer.train(); trainer.save_model(out/'best'); processor.save_pretrained(out/'best'); return
    pred=[]
    for i in range(0,len(ds),batch_size):
        b=collate([ds[j] for j in range(i,min(i+batch_size,len(ds)))])
        with torch.no_grad(): ids=model.generate(b['input_features'].to(model.device))
        pred.extend(processor.batch_decode(ids,skip_special_tokens=True))
    refs=[r['text'] for r in rows]
    try:
        from jiwer import wer, cer
        metrics={'wer':wer(refs,pred),'cer':cer(refs,pred)}
    except Exception as e: metrics={'metric_error':str(e)}
    by={}
    for r,p in zip(rows,pred): by.setdefault(r.get('dialect',''),[]).append((r['text'],p))
    for d,pairs in by.items():
        try: metrics[f'{d}_wer']=wer([x[0] for x in pairs],[x[1] for x in pairs]); metrics[f'{d}_cer']=cer([x[0] for x in pairs],[x[1] for x in pairs])
        except Exception: pass
    json.dump(metrics,open(out/'metrics.json','w'),ensure_ascii=False,indent=2); json.dump([{'reference':r,'prediction':p} for r,p in zip(refs,pred)],open(out/'predictions.json','w'),ensure_ascii=False,indent=2)

def main_eval():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True); p.add_argument('--model',default='openai/whisper-small'); p.add_argument('--batch-size',type=int,default=2); a=p.parse_args(); score(a.model,a.manifest,a.output,a.batch_size)
def main_train():
    p=argparse.ArgumentParser(); p.add_argument('--train',required=True); p.add_argument('--validation',required=False); p.add_argument('--output',required=True); p.add_argument('--model',default='openai/whisper-small'); p.add_argument('--max-steps',type=int,default=1000); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--gradient-accumulation',type=int,default=8); p.add_argument('--fp16',action='store_true'); p.add_argument('--gradient-checkpointing',action='store_true'); p.add_argument('--max-train-items',type=int); a=p.parse_args(); score(a.model,a.train,a.output,a.batch_size,a.max_steps,True,a.max_train_items,a.fp16,a.gradient_checkpointing)
def main_infer():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--audio',required=True); p.add_argument('--output',required=True); a=p.parse_args(); from transformers import pipeline; out=pipeline('automatic-speech-recognition',model=a.model,chunk_length_s=30,device=0 if __import__('torch').cuda.is_available() else -1)(a.audio,generate_kwargs={'language':'am','task':'transcribe'}); json.dump(out,open(a.output,'w'),ensure_ascii=False,indent=2)
