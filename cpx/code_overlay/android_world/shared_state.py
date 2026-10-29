"""Host-local shared state for workers mounting the same filesystem.

Locks live beside data files and survive atomic replacement. Keep critical
sections limited to file IO: never hold a state lock during inference or ADB.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def lock(path, exclusive=True, blocking=True):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.with_name(path.name + '.lock').open('a+b') as handle:
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if not blocking:
      mode |= fcntl.LOCK_NB
    fcntl.flock(handle.fileno(), mode)
    try:
      yield
    finally:
      fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_text(path, text):
  """Caller owns the corresponding lock for read-modify-write operations."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
  try:
    os.fchmod(fd, 0o644)
    with os.fdopen(fd, 'w') as handle:
      handle.write(text)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def read_text(path, default=''):
  with lock(path, exclusive=False):
    return Path(path).read_text() if Path(path).exists() else default


def read_json(path, default):
  text = read_text(path)
  return json.loads(text) if text else default


def update_json(path, update, default):
  with lock(path):
    p = Path(path)
    current = json.loads(p.read_text()) if p.exists() else default
    result = update(current)
    atomic_text(p, json.dumps(result, ensure_ascii=False))
    return result


def append_text(path, text):
  if not text:
    return
  with lock(path):
    with Path(path).open('a') as handle:
      handle.write(text)
      handle.flush()
      os.fsync(handle.fileno())


def append_json(path, row):
  append_text(path, json.dumps(row, ensure_ascii=False) + '\n')


def read_delta(path, offset=0):
  """Return complete committed records, without consuming an incomplete tail."""
  with lock(path, exclusive=False):
    p = Path(path)
    if not p.exists():
      if offset:
        raise RuntimeError(f'Shared log disappeared: {p}')
      return [], 0
    if p.stat().st_size < offset:
      raise RuntimeError(f'Shared log was truncated during the run: {p}')
    with p.open('rb') as handle:
      handle.seek(offset)
      rows = []
      while True:
        start = handle.tell()
        line = handle.readline()
        if not line or not line.endswith(b'\n'):
          return rows, start
        if line.strip():
          rows.append(json.loads(line))


def rows(path):
  return read_delta(path)[0]


def append_wait(path, key, value):
  def update(data):
    data[key] = (data.get(key, []) + [value])[-9:]
    return data
  return update_json(path, update, {})


def episode_path(raw, task, variant):
  # Warm evaluation retains the source episode's captures separately from the
  # current episode; identical task/capture names must never overwrite them.
  mapping_path = os.environ.get('MAI_EPISODE_ROOTS_JSON')
  if mapping_path:
    sources = json.loads(Path(mapping_path).read_text())
    if variant in sources:
      source = sources[variant]
      return Path(source['raw']) / task / source.get('episode', 'p0')
  # A shared run has one immutable episode directory per task, visible to peers.
  return Path(raw) / task / ('p0' if os.environ.get('MAI_SHARED_STATE') == '1' else variant.split('_')[0])
