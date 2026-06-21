import platform
import ctypes
import winreg

def get_hardware_info() -> dict:
    """
    Retrieves host machine specifications (OS, CPU, total memory) using native 
    APIs and registry checks, avoiding external library dependencies.
    """
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "cpu": platform.processor(),
        "ram": "Unknown",
        "python_version": platform.python_version()
    }
    
    if platform.system() == "Windows":
        # Get RAM details via GlobalMemoryStatusEx
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram"] = f"{round(stat.ullTotalPhys / (1024**3), 2)} GB"
        except Exception:
            pass
            
        # Get actual processor brand name from Windows Registry
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            info["cpu"] = cpu_name.strip()
        except Exception:
            pass
            
    return info

if __name__ == "__main__":
    print(get_hardware_info())
