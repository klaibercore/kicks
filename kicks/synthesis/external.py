"""Supervise optional upstream models in their own Python environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from ..analysis.benchmark import audit_directory,file_hash,identity,write_json
from ..instruments import get_profile


def external_generate(spec,*,out,python=sys.executable,timeout=1800):
    # Shared validation imports no optional providers and changes no environment
    # settings. Provider loading and offline settings run only in the child.
    job=json.loads(Path(spec).read_text())
    from ..external.worker import validate_job
    validate_job(job)
    profile=get_profile(job.get('instrument','kick'))
    checkpoint=Path(job.get('checkpoint','')).resolve()
    if not checkpoint.exists():raise FileNotFoundError('local external weights are required')
    directory=Path(out)
    if directory.exists():raise FileExistsError('external output directory must be new')
    candidates=[checkpoint] if checkpoint.is_file() else sorted(p for p in checkpoint.rglob('*')
        if p.is_file() and p.suffix in ('.bin','.safetensors','.pth','.pt','.ckpt','.json'))
    if not candidates:raise ValueError('model directory contains no recognized weights/configuration')
    identities=[{'path':str(p.relative_to(checkpoint) if checkpoint.is_dir() else p.name),'sha256':file_hash(p)} for p in candidates]
    config=job.get('config')
    if config:
        identities.append({'path':'model_config','sha256':file_hash(config)})
    if job.get('source'):identities.append({'path':'source','sha256':file_hash(job['source'])})
    job={**job,'checkpoint':str(checkpoint),'out':str(directory.resolve())}
    directory.mkdir(parents=True)
    write_json(directory/'job.json',job)
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
    worker=Path(__file__).resolve().parents[1]/'external/worker.py'
    result=subprocess.run([python,str(worker),str(directory/'job.json')],env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    write_json(directory/'execution.json',{'returncode':result.returncode,'stdout':result.stdout[-4000:],
        'stderr':result.stderr[-4000:],'identities':identities,'model_identity':identity(identities)})
    if result.returncode:
        raise RuntimeError(f'external worker failed; diagnostics retained in {directory}/execution.json')
    report=audit_directory(directory/'conformed',profile.name,out=directory/'audit',backend=job['kind'])
    write_json(directory/'report.json',{'kind':job['kind'],'model_identity':identity(identities),
        'prompt':job.get('prompt'),'numerical_descriptor_conditioning':False,'held_out':False,
        'reconstruction_source':job.get('source'),'listening_judged':False,'summary':report['summary']})
    return report
