"""Download the pinned official CUDA wheel in resumable verified chunks."""
from download_models import ROOT, download_file

NAME = 'torch-2.10.0+cu128-cp312-cp312-win_amd64.whl'
SHA256 = 'fbde8f6a9ec8c76979a0d14df21c10b9e5cab6f0d106a73ca73e2179bc597cae'

if __name__ == '__main__':
    download_file('pytorch', SHA256, {
        'rfilename': NAME,
        'size': 2867409626,
        'lfs': {'sha256': SHA256},
        '_url': 'https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-win_amd64.whl',
        '_workers': 8,
    }, ROOT / '.cache' / 'wheels')
    print('CUDA wheel ready.', flush=True)
