"""Opt-in Windows loader setup for NVIDIA's separately installed CUDA wheels."""
import importlib.metadata
import hashlib
import os
from pathlib import Path

PACKAGES=('nvidia-cublas-cu12','nvidia-cudnn-cu12','nvidia-cuda-runtime-cu12','nvidia-cuda-nvrtc-cu12')

def register_nvidia_wheels():
    """Return live DLL-directory handles and version metadata; caller retains handles.

    This changes only this process, not the user's global PATH. The driver remains
    a separate prerequisite. Do not mix this mode with arbitrary vendor folders.
    """
    if os.name!='nt':raise ValueError('Windows NVIDIA-wheel loader selected on another platform')
    directories=set();versions={}
    for name in PACKAGES:
        distribution=importlib.metadata.distribution(name);versions[name]=distribution.version
        found=False
        for file in distribution.files or ():
            if str(file).lower().endswith('.dll'):
                path=Path(distribution.locate_file(file)).resolve(strict=True)
                directories.add(path.parent);found=True
        if not found:raise ValueError('Installed NVIDIA package has no DLL files: '+name)
    paths=sorted(directories,key=str)
    handles=[os.add_dll_directory(str(path)) for path in paths]
    os.environ['PATH']=os.pathsep.join(map(str,paths))+os.pathsep+os.environ.get('PATH','')
    return handles,versions

def verify_loaded_nvidia_wheels():
    """Verify loaded CUDA DLLs belong to declared wheels, without publishing paths.

    CTranslate2's Windows wheel can bundle cuDNN itself. Its own package directory
    takes part in DLL resolution and is explicitly included in this check.
    """
    import psutil
    allowed={}
    for name in (*PACKAGES,'ctranslate2'):
        distribution=importlib.metadata.distribution(name)
        for file in distribution.files or ():
            if str(file).lower().endswith('.dll'):
                allowed[str(Path(distribution.locate_file(file)).resolve()).casefold()]=name
    result=[]
    for mapping in psutil.Process().memory_maps():
        path=Path(mapping.path)
        if path.name.casefold().startswith(('cublas','cudnn','cudart')) and path.suffix.casefold()=='.dll':
            package=allowed.get(str(path.resolve()).casefold())
            if package is None:raise ValueError('A CUDA math library was loaded from outside the declared runtime packages')
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            result.append({'file':path.name,'package':package,'sha256':h.hexdigest()})
    for prefix in ('cublas','cudnn'):
        if not any(r['file'].casefold().startswith(prefix) for r in result):raise ValueError('No loaded '+prefix+' library observed')
    return result
