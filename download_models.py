"""One-time download. Runtime inference never connects to the internet."""
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'

MODEL_REPO = 'hlhc/FLUX.2-klein-4B-bnb-4bit'
MODEL_REVISION = '199c1411ab9dc4111e9fe7e8e3dc3fc4e63c2648'
LORA_REPO = 'Alissonerdx/BFS-Best-Face-Swap'
LORA_NAME = 'bfs_head_v1_flux-klein_4b.safetensors'


CHUNK = 8 * 1024 * 1024


@contextmanager
def model_download_lock():
    """Two installer windows must not assemble/delete the same chunks."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    key = hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:16]
    handle = kernel.CreateMutexW(None, False, 'Local\\HeadSwapModels_' + key)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        print('Preparing model download (waiting if another installer is active)...', flush=True)
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) not in (0, 128):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            kernel.ReleaseMutex(handle)
    finally:
        kernel.CloseHandle(handle)


def get_metadata(repo):
    with urllib.request.urlopen('https://huggingface.co/api/models/' + repo + '?blobs=true', timeout=30) as response:
        return json.load(response)


def download_file(repo, revision, entry, folder):
    """Resume verified-size chunks. Validate the final safetensors SHA-256."""
    name, size = entry['rfilename'], entry['size']
    destination = folder / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        return
    url = entry.get('_url') or f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
    if size < CHUNK:
        with urllib.request.urlopen(url, timeout=90) as response:
            data = response.read()
        if len(data) != size:
            raise RuntimeError(f'Incomplete file: {name}')
        temp = destination.with_suffix(destination.suffix + '.incomplete')
        temp.write_bytes(data)
        temp.replace(destination)
        return

    parts = ROOT / 'models' / '.downloads' / hashlib.sha256((repo + revision + name).encode()).hexdigest()[:24]
    parts.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(start + CHUNK, size) - 1) for start in range(0, size, CHUNK)]
    completed = 0

    def fetch(bounds):
        start, end = bounds
        path = parts / f'{start:012d}.part'
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return expected
        # A distinct resolve URL avoids proxies caching a different byte range.
        request = urllib.request.Request(url + f'?download=true&part={start}',
                                         headers={'Range': f'bytes={start}-{end}'})
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    content_range = response.headers.get('Content-Range', '')
                    if response.status != 206 or not content_range.startswith(f'bytes {start}-{end}/'):
                        raise RuntimeError(f'Server did not honor byte range for {name}')
                    data = response.read(expected + 1)
                if len(data) != expected:
                    raise RuntimeError(f'Incomplete chunk for {name}')
                path.write_bytes(data)
                return expected
            except (OSError, RuntimeError):
                if attempt:
                    raise

    print(f'Downloading {name}: {size / 2**30:.2f} GiB (resumable)', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=entry.get('_workers', 4)) as pool:
        futures = [pool.submit(fetch, bounds) for bounds in ranges]
        last_report = -1
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            percent = int(completed * 100 / size)
            if percent // 5 != last_report:
                last_report = percent // 5
                print(f'  {name}: {percent}% ({completed / 2**20:.0f} MiB)', flush=True)
    temp = destination.with_suffix(destination.suffix + '.incomplete')
    digest = hashlib.sha256()
    with temp.open('wb') as output:
        for start, _ in ranges:
            with (parts / f'{start:012d}.part').open('rb') as part:
                while block := part.read(4 * 1024 * 1024):
                    digest.update(block)
                    output.write(block)
    expected_hash = entry.get('lfs', {}).get('sha256') or entry.get('lfs', {}).get('oid')
    if expected_hash and digest.hexdigest() != expected_hash:
        raise RuntimeError(f'Checksum mismatch: {name}. The file has not been activated.')
    temp.replace(destination)
    # Delete only known chunk files inside this application's download directory.
    if not parts.resolve().is_relative_to((ROOT / 'models').resolve()):
        raise RuntimeError('Unexpected download directory')
    for start, _ in ranges:
        (parts / f'{start:012d}.part').unlink()
    parts.rmdir()


def configure_for_turing(model_path):
    # Turing (RTX 20xx) has no native BF16. Keep NF4 weights but compute in FP16.
    for name in ('transformer', 'text_encoder'):
        path = model_path / name / 'config.json'
        config = json.loads(path.read_text(encoding='utf-8'))
        config['quantization_config']['bnb_4bit_compute_dtype'] = 'float16'
        if name == 'text_encoder':
            config['dtype'] = 'float16'
        else:
            config.pop('dtype', None)  # Diffusers transformer has no dtype config field.
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')


def main():
    (ROOT / 'models').mkdir(exist_ok=True)
    if (ROOT / 'models' / 'ready.json').exists():
        manifest = json.loads((ROOT / 'models' / 'ready.json').read_text(encoding='utf-8'))
        if all((ROOT / 'models' / name).exists() for name in manifest.get('files', [])) and manifest.get('files'):
            configure_for_turing(ROOT / 'models' / 'klein')
            print('Models already installed.', flush=True)
            return
    model_path = ROOT / 'models' / 'klein'
    # Fetch the pinned revision's file metadata, not a changing branch head.
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true', timeout=30) as response:
        metadata = json.load(response)
    files = []
    for entry in metadata['siblings']:
        name = entry['rfilename']
        if (name in ('model_index.json', 'LICENSE.md') or
                name.startswith(('scheduler/', 'tokenizer/', 'text_encoder/', 'transformer/', 'vae/'))):
            download_file(MODEL_REPO, MODEL_REVISION, entry, model_path)
            files.append('klein/' + name)
    configure_for_turing(model_path)
    metadata = get_metadata(LORA_REPO)
    revision = metadata['sha']
    entry = next(item for item in metadata['siblings'] if item['rfilename'] == LORA_NAME)
    download_file(LORA_REPO, revision, entry, ROOT / 'models' / 'bfs')
    files.append('bfs/' + LORA_NAME)
    manifest = {'model': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'lora': LORA_REPO, 'lora_revision': revision, 'lora_file': LORA_NAME,
                'files': files}
    (ROOT / 'models' / 'ready.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print('Models ready. Photos are processed entirely offline.')


if __name__ == '__main__':
    with model_download_lock():
        main()
