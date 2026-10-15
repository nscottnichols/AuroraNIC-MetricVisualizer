"""Safe server-side trace discovery, provenance defaults, and cache identities."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib

# Cache identity covers collector attribution and explicit sampling inventories.
PIPELINE_VERSION = 'nicviz-input/1:process-thread:explicit-sampling-inventory'
TRACE_SUFFIXES = {'.pftrace','.pb','.perfetto-trace'}


def resolve_trace_path(root, value):
    base=Path(root).expanduser().resolve(strict=True)
    if not isinstance(value,str) or not value.strip():raise ValueError('Enter a trace path or choose an entry below.')
    path=Path(value).expanduser()
    if not path.is_absolute():path=base/path
    path=path.resolve(strict=True)
    if not path.is_relative_to(base):raise ValueError('Path is outside the configured browse root. Restart with --browse-root to allow another directory.')
    if not (path.is_file() or path.is_dir()):raise ValueError('Select a regular trace file or directory.')
    return path


def infer_format(path):
    path=Path(path)
    if path.is_dir() or path.name=='metadata':return 'ctf'
    if path.suffix.lower() in TRACE_SUFFIXES:return 'pftrace'
    raise ValueError('Choose a CTF directory (or metadata file) or a binary .pftrace/.pb file. Extract archives first.')


def trace_fingerprint(path, cancellation=None):
    path=Path(path).resolve(strict=True)
    digest=hashlib.sha256(PIPELINE_VERSION.encode())
    root=path if path.is_dir() else path.parent
    paths=sorted(path.rglob('*')) if path.is_dir() else [path]
    count=0
    for file in paths:
        if any(part in {'.nicviz-cache','__pycache__','.git'} for part in file.relative_to(root).parts):continue
        if file.is_symlink():raise ValueError('Trace directories containing symlinks are not imported; choose the resolved trace directory.')
        if not file.is_file():continue
        count+=1
        if count>100_000:raise ValueError('Trace directory contains too many files; choose the CTF subdirectory, not all runs.')
        digest.update(str(file.relative_to(root)).encode());digest.update(b'\0')
        with file.open('rb') as f:
            while block:=f.read(1024*1024):
                if cancellation:cancellation.check('source fingerprint')
                digest.update(block)
        digest.update(b'\0')
    return digest.hexdigest()


def trace_arguments(path, *, root=None, format='auto', cache=True):
    path=Path(path).expanduser().resolve(strict=True)
    if path.name=='metadata':path=path.parent
    fmt=infer_format(path) if format=='auto' else format
    if fmt not in {'ctf','pftrace'}:raise ValueError('Unknown trace format.')
    if fmt == 'pftrace' and not path.is_file():raise ValueError('Choose a pftrace file, not a directory.')
    if fmt == 'ctf' and not path.is_dir():raise ValueError('Choose a CTF directory or its metadata file.')
    base=Path(root).resolve() if root is not None else path.parent
    # Hashing happens in the cancellable loader, not the web callback.
    path_key=hashlib.sha256((str(path)+PIPELINE_VERSION).encode()).hexdigest()[:24]
    return SimpleNamespace(trace=str(path),trace_format=fmt,
        trace_run_id=path.parent.name if path.name in {'ctf','timeline.pftrace'} else path.stem,
        trace_source_fingerprint='auto', trace_reader_version='auto',
        trace_clock_domain='CTF metadata clock' if fmt=='ctf' else 'THAPI Perfetto timestamp',
        trace_synchronization_evidence='Cross-host synchronization not independently established.',
        trace_requested_counter_file=None,
        trace_cache=str(base/'.nicviz-cache'/f'{path_key}.npz') if cache else None,
        trace_ctf_reader='auto')
