"""Verify every archived raw file against its server-side SHA256 manifest."""
from pathlib import Path
import hashlib,json,tarfile
r=Path(__file__).resolve().parent
for model in ['8B','2B']:
 manifest=json.loads((r/(model+'_files.json')).read_text());entries={x['path']:x for x in manifest['files'] if x['in_archive']};checked=set()
 with tarfile.open(r/(model+'_logs.tar.gz')) as t:
  for member in t:
   if not member.isfile() or member.name.startswith('actual_launch/'):continue
   rel=member.name.split('/',1)[1];assert rel in entries,member.name
   data=t.extractfile(member).read();e=entries[rel];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'],rel;checked.add(rel)
 assert checked==set(entries),(model,set(entries)-checked)
 print(model, len(checked),'raw files verified; screenshots retained at',manifest['server_root'])
