from pathlib import Path

def get_logdir(path):
    path = Path(path)
    candidate = path.parent / f'{path.name}_0/'
    if not candidate.exists():
        return candidate
    num = max(
        [int(f.name.split('_')[-1]) for f in path.parent.iterdir() if f.name.startswith(path.name)]
    ) + 1
    return path.parent / f'{path.name}_{num}/'

def copy_files(files, path):
    path = Path(path)
    for f in files:
        if isinstance(f, list):
            name = Path(f[1]).name
            content = Path(f[0]).read_text()
        else:
            name = Path(f).name
            content = Path(f).read_text()
        with open(path/name, 'w') as dest:
            dest.write(content)
        (path/name).chmod(0o444)
