from pathlib import Path
import importlib.util
import platform
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"

print("=" * 60)
print("GenRec Environment Check")
print("=" * 60)

print(f"project_root: {PROJECT_ROOT}")
print(f"python_executable: {sys.executable}")
print(f"python_version: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")

print("\n[Python packages]")

packages = {
    "torch": "torch",
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "tensorboard": "tensorboard",
    "RecBole": "recbole",
}

for display_name, module_name in packages.items():
    installed = importlib.util.find_spec(module_name) is not None
    status = "OK" if installed else "MISSING"
    print(f"{display_name}: {status}")

print("\n[PyTorch and CUDA]")

try:
    import torch

    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_memory = properties.total_memory / (1024 ** 3)

            print(f"cuda_device_{index}: {properties.name}")
            print(f"cuda_device_{index}_memory_gb: {total_memory:.2f}")
            print(
                f"cuda_device_{index}_capability: "
                f"{properties.major}.{properties.minor}"
            )
    else:
        print("cuda_device: CPU mode")

except Exception as error:
    print(f"torch_check_error: {error}")

print("\n[Project files]")

print(f"base_config_exists: {CONFIG_PATH.exists()}")

if CONFIG_PATH.exists():
    try:
        import yaml

        with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
            config = yaml.safe_load(file)

        print("base_config_valid: True")
        print(f"project_name: {config.get('project_name')}")
        print(f"dataset: {config.get('dataset')}")
        print(f"device: {config.get('device')}")

    except Exception as error:
        print(f"base_config_valid: False")
        print(f"config_error: {error}")

print("=" * 60)
