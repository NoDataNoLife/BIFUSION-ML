"""Manual RunPod start/status/job helper. No credentials are embedded or printed."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path('/workspace/bifusion-ml')
PYTHON = Path('/workspace/bifusion-venv/bin/python')
CELERY = 'app.core.celery_app.celery_app'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=2):
            return True
    except OSError:
        return False


def api(path, settings):
    request = Request('http://127.0.0.1:8000' + path,
                      headers={'X-API-Key': settings.API_KEY})
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.loads(error.read().decode())


def workers():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().decode().split('\0')
            if ('celery' in args and 'worker' in args and CELERY in args
                    and (entry / 'cwd').resolve() == ROOT.resolve()):
                found.append(int(entry.name))
        except (OSError, UnicodeError):
            pass
    return found


def launch(args, log):
    with open('/workspace/' + log, 'ab', buffering=0) as output:
        output.write(('\n--- start ' + time.strftime('%Y-%m-%d %H:%M:%S') + ' ---\n').encode())
        return subprocess.Popen([str(PYTHON)] + args, cwd=ROOT,
                                stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)


def wait_api(process, settings):
    for _ in range(60):
        require(process is None or process.poll() is None,
                'API exited. Read /workspace/uvicorn-current.log')
        try:
            code, body = api('/health', settings)
            if code == 200 and body.get('status') == 'ok':
                return
        except (OSError, URLError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError('API not ready after 120 seconds. Read /workspace/uvicorn-current.log')


def status(settings):
    from redis import Redis
    try:
        redis = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=3, socket_timeout=3)
        print('Redis:', 'PONG' if redis.ping() else 'FAILED')
        for queue in ('augmentation', 'training', 'inference'):
            print('Queue', queue + ':', redis.llen(queue))
    except Exception as error:
        print('Redis error:', type(error).__name__)
    try:
        code, body = api('/health', settings)
        print('API health:', code, body)
        code, body = api('/api/v1/augmentation/ops-check-nonexistent', settings)
        print('Authenticated lookup:', code, body)
    except (OSError, URLError, ValueError) as error:
        print('API error:', type(error).__name__)
    print('Worker process IDs:', workers(), '(process existence only; not model readiness)')
    pod = os.environ.get('RUNPOD_POD_ID')
    if pod:
        print('ML_BASE_URL: https://' + pod + '-8000.proxy.runpod.net')
    print('S3 augmentation enabled:', settings.AUGMENTATION_PUBLISH_S3)
    print('S3 training enabled:', getattr(settings, 'TRAINING_PUBLISH_S3', False))
    print('S3 output prefix:', settings.S3_OUTPUT_PREFIX)
    print('Health/auth checks do not verify model execution or file downloads.')


def start(settings, gpu, resume_queued=False):
    from urllib.parse import urlsplit
    from redis import Redis
    require(bool(settings.API_KEY), 'API_KEY is empty; restore the agreed key in .env.')
    if gpu:
        import torch
        require(torch.cuda.is_available(), 'No GPU detected. Use CPU mode only for API checks.')
        require(settings.DEVICE in ('auto', 'cuda'), 'DEVICE must be auto or cuda.')
        for path in (Path(settings.BIFUSION_PACKAGE_PATH), Path(settings.MEDFUSION_CODE_PATH),
                     Path(settings.MEDFUSION_WEIGHTS_DIR) / settings.MEDFUSION_DIFFUSION_CKPT,
                     Path(settings.MEDFUSION_WEIGHTS_DIR) / settings.MEDFUSION_VAE_CKPT):
            require(path.exists(), 'Missing configured model path: ' + str(path))
        print('GPU:', torch.cuda.get_device_name(0))
    redis = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=3, socket_timeout=3)
    try:
        redis.ping()
    except Exception:
        target = urlsplit(settings.REDIS_URL)
        require(target.hostname in ('localhost', '127.0.0.1')
                and target.port == 6379 and not target.password,
                'Configured Redis unavailable; this helper only starts local unauthenticated Redis:6379.')
        require(not listening(6379), 'Port 6379 occupied but configured Redis cannot be reached.')
        if not shutil.which('redis-server'):
            require(os.geteuid() == 0, 'Redis missing; install redis-server as root.')
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 'redis-server'], check=True)
        Path('/workspace/redis').mkdir(exist_ok=True)
        subprocess.run(['redis-server', '--bind', '127.0.0.1', '--protected-mode', 'yes',
                        '--port', '6379', '--dir', '/workspace/redis', '--appendonly', 'yes',
                        '--daemonize', 'yes'], check=True)
        redis.ping()
    process = None
    if not listening(8000):
        process = launch(['-u', '-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0',
                          '--port', '8000'], 'uvicorn-current.log')
    wait_api(process, settings)
    code, body = api('/api/v1/augmentation/ops-check-nonexistent', settings)
    require(code == 404 and body.get('detail', {}).get('error_code') == 'JOB_NOT_FOUND',
            'API auth/store check failed. Run status and inspect API log; no worker started.')
    if gpu and not workers():
        counts = {q: redis.llen(q) for q in ('augmentation', 'training', 'inference')}
        require(resume_queued or not any(counts.values()),
                'Existing queued jobs found: ' + str(counts)
                + '. Review them; use start --gpu --resume-queued to process them.')
        if any(counts.values()):
            print('Processing existing queued jobs:', counts)
        worker = launch(['-m', 'celery', '-A', CELERY, 'worker', '--pool=solo', '--concurrency=1',
                         '-Q', 'augmentation,training,inference', '--loglevel=info',
                         '--hostname=bifusion-test@%h'], 'celery-test.log')
        from app.core.celery_app import celery_app
        ready = False
        for _ in range(15):
            require(worker.poll() is None, 'Worker exited. Read /workspace/celery-test.log')
            if celery_app.control.ping(timeout=2):
                ready = True
                break
            time.sleep(2)
        require(ready, 'Worker did not reply. Read /workspace/celery-test.log; do not start duplicates.')
    elif gpu:
        print('Existing worker retained. Check logs; it may be busy or using earlier .env settings.')
    elif workers():
        print('WARNING: an existing worker is running. CPU mode does not stop it.')
    print('START CHECKS PASSED. Existing processes were not restarted; .env changes need a controlled restart.')
    status(settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'job'))
    parser.add_argument('job_id', nargs='?')
    parser.add_argument('--gpu', action='store_true', help='Validate GPU and start a serial model worker.')
    parser.add_argument('--resume-queued', action='store_true',
                        help='Allow the GPU worker to process existing queued requests.')
    parser.add_argument('--kind', choices=('augmentation', 'training', 'inference'), default='augmentation')
    args = parser.parse_args()
    require(sys.platform == 'linux' and ROOT.is_dir() and PYTHON.exists(),
            'Run this file in the RunPod web terminal with /workspace/bifusion-venv/bin/python.')
    require(Path(sys.prefix).resolve() == PYTHON.parent.parent.resolve(), 'Use the RunPod virtualenv Python.')
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from app.core.config import settings
    if args.action == 'start':
        import fcntl
        with open('/workspace/.bifusion-start.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            start(settings, args.gpu, args.resume_queued)
    elif args.action == 'status':
        status(settings)
    else:
        require(bool(args.job_id), 'Supply the ML job_id, not the backend job ID.')
        code, body = api('/api/v1/' + args.kind + '/' + quote(args.job_id, safe=''), settings)
        print('HTTP', code)
        print(json.dumps(body, ensure_ascii=False, indent=2))
        print('Result may contain temporary signed download URLs; avoid sharing them publicly.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOP:', type(error).__name__, str(error))
        sys.exit(1)
