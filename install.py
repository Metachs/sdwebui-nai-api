import launch


def Installed(l):
    if not launch.is_installed(l): return False
    return True  

if not Installed("requests"): launch.run_pip("install requests", desc="install requests")
