"""Replay screenshots as simultaneous MAI requests; never execute device actions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time

from PIL import Image
import requests
from mai_client import OfficialClient
from mai_protocol import NativeHistory, parse_response


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--screens', type=Path, required=True)
  ap.add_argument('--output', type=Path, required=True)
  ap.add_argument('--label', required=True)
  ap.add_argument('--requests', type=int, default=8)
  ap.add_argument('--images', type=int, choices=[1, 3], default=3)
  args = ap.parse_args()
  config_dir = Path(__file__).resolve().parents[1] / 'configs'
  config = json.loads((config_dir / 'mai_ui_8b_official.json').read_text())
  prompt = (config_dir / config['system_prompt_file']).read_text().rstrip('\n')
  paths = sorted(p for p in args.screens.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
  if len(paths) < args.requests:
    raise RuntimeError('Need distinct screenshots for each replay request')
  payloads = []
  for i in range(args.requests):
    images = [Image.open(paths[(i + j) % len(paths)]).convert('RGB') for j in range(args.images)]
    hist = NativeHistory(3)
    for j, img in enumerate(images[:-1]):
      hist.append({'action': 'wait'}, f'Observe screen {j} before choosing the next action.', img)
    goal = f'Replay case {i}: inspect the current screen and choose the next action to create a recipe in the Broccoli app.'
    messages = hist.messages(prompt, goal, images[-1])
    payloads.append(OfficialClient(config['api_url'], config['model'], config).payload(messages))
  base = config['api_url'].split('/v1/')[0]
  samples, rows = [], []
  stop = threading.Event()
  barrier = threading.Barrier(args.requests)
  def sample():
    session = requests.Session()
    session.trust_env = False
    while not stop.is_set():
      try:
        body = session.get(base + '/metrics', timeout=3).text
        row = {'t': time.monotonic()}
        for line in body.splitlines():
          if line.startswith('#'):
            continue
          for name in ['num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc', 'num_preemptions_total']:
            if re.match(r'vllm:' + name + r'(\{|\s)', line):
              row[name] = row.get(name, 0) + float(line.rsplit(' ', 1)[1])
        samples.append(row)
      except Exception as e:
        samples.append({'monitor_error': repr(e)})
      stop.wait(0.25)
  def request(i):
    session = requests.Session()
    session.trust_env = False
    barrier.wait(timeout=30)
    t0 = time.monotonic()
    row = {'case': i}
    try:
      response = session.post(config['api_url'], json=payloads[i], timeout=180)
      response.raise_for_status()
      result = response.json()
      choice = result['choices'][0]
      if choice.get('finish_reason') == 'length':
        raise RuntimeError('Truncated response')
      parsed = parse_response(choice['message']['content'], (1080, 2400))
      row.update(ok=True, usage=result.get('usage'), action=parsed['native_arguments'])
    except Exception as e:
      row.update(ok=False, error=repr(e))
    row['latency_s'] = time.monotonic() - t0
    return row
  monitor = threading.Thread(target=sample)
  monitor.start()
  t0 = time.monotonic()
  with ThreadPoolExecutor(max_workers=args.requests) as pool:
    rows = list(pool.map(request, range(args.requests)))
  elapsed = time.monotonic() - t0
  stop.set()
  monitor.join(timeout=5)
  summary = {'label': args.label, 'requests': args.requests, 'images_per_request': args.images,
             'successful': sum(r['ok'] for r in rows), 'batch_wall_s': elapsed,
             'requests_per_minute': sum(r['ok'] for r in rows) * 60 / elapsed,
             'latency_mean_s': statistics.mean(r['latency_s'] for r in rows),
             'latency_max_s': max(r['latency_s'] for r in rows),
             'peak_running': max((r.get('num_requests_running', 0) for r in samples), default=0),
             'peak_waiting': max((r.get('num_requests_waiting', 0) for r in samples), default=0),
             'peak_kv_usage': max((r.get('kv_cache_usage_perc', 0) for r in samples), default=0),
             'preemptions_delta': max((r.get('num_preemptions_total', 0) for r in samples), default=0) - min((r.get('num_preemptions_total', 0) for r in samples), default=0),
             'payload_sha256': [hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest() for p in payloads],
             'results': rows, 'metrics': samples}
  args.output.write_text(json.dumps(summary, indent=2))
  print(json.dumps({k: v for k, v in summary.items() if k not in {'results', 'metrics', 'payload_sha256'}}), flush=True)
  if summary['successful'] != args.requests:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
