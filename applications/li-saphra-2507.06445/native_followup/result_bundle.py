"""Pack raw result trees losslessly, or restore them for the frozen analyzers.

The archive preserves every file's bytes; the index lists their hashes. It is
storage packaging, not a new producer or a replacement for run provenance.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT=Path(__file__).resolve().parent


def sha(data):return hashlib.sha256(data).hexdigest()


def pack(names, archive, index):
    if archive.exists() or index.exists():raise ValueError('Refusing overwrite')
    archive.parent.mkdir(parents=True,exist_ok=True)
    files={}
    with archive.open('xb') as handle, gzip.GzipFile(fileobj=handle,mode='wb',mtime=0,filename='') as gz, tarfile.open(fileobj=gz,mode='w|') as tar:
        for name in names:
            directory=ROOT/'results'/name
            for path in sorted(directory.rglob('*')):
                if not path.is_file():continue
                relative=path.relative_to(ROOT/'results').as_posix()
                data=path.read_bytes()
                entry=tarfile.TarInfo(relative);entry.size=len(data);entry.mode=0o644;entry.mtime=0
                tar.addfile(entry,io.BytesIO(data))
                files[relative]={'sha256':sha(data),'bytes':len(data)}
    index.write_text(json.dumps({'archive_sha256':sha(archive.read_bytes()),'files':files},indent=2)+'\n')
    print('Packed',len(files),'files;',archive.stat().st_size,'compressed bytes')


def unpack(archive,index,destination):
    expected=json.loads(index.read_text())
    if sha(archive.read_bytes())!=expected['archive_sha256']:raise ValueError('Archive checksum mismatch')
    seen=set()
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            relative=PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts or not member.isfile() or member.name not in expected['files'] or member.name in seen:
                raise ValueError('Unexpected archive member')
            data=tar.extractfile(member).read()
            info=expected['files'][member.name]
            if sha(data)!=info['sha256'] or len(data)!=info['bytes']:raise ValueError('Result file checksum mismatch')
            target=destination.joinpath(*relative.parts)
            if target.exists() and target.read_bytes()!=data:raise ValueError('Refusing to replace differing local result')
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(data)
            seen.add(member.name)
    if seen!=set(expected['files']):raise ValueError('Incomplete archive')
    print('Verified/restored',len(seen),'files')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['pack','unpack'])
    p.add_argument('--archive',type=Path,default=ROOT/'artifacts/confirmation_001.tar.gz')
    p.add_argument('--index',type=Path,default=ROOT/'artifacts/confirmation_001.index.json')
    p.add_argument('--destination',type=Path,default=ROOT/'results')
    p.add_argument('--names',nargs='+',default=['confirmation_001_continued'])
    a=p.parse_args()
    if a.action=='pack':pack(a.names,a.archive,a.index)
    else:unpack(a.archive,a.index,a.destination)
