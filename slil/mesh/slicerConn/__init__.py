

def _getSlicerInstallPath():
    from pathlib import Path
    from os import environ, listdir
    app_data_local = Path(environ['USERPROFILE'], 'AppData', 'Local')
    slicer_install_path = app_data_local.joinpath('slicer.org')
    if not slicer_install_path.exists():
        slicer_install_path = app_data_local.joinpath('NA-MIC')
        if not slicer_install_path.exists():
            raise FileNotFoundError(f'3D Slicer not found in {app_data_local.joinpath("slicer.org")} or {app_data_local.joinpath("NA-MIC")}')

    available = listdir(slicer_install_path)
    latest_version = available[-1]
    return slicer_install_path.joinpath(latest_version)