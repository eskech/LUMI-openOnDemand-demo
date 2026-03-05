"""
Register qwen3_5_moe with transformers 4.52.4 at runtime.

qwen3_5_moe was added to transformers after 4.52.4, and the model's
config.json has no auto_map, so trust_remote_code=True cannot help.

Strategy:
  1. Try auto_map in config.json  (fast path for models that have it)
  2. Fall back to snapshot_download for *.py files from the repo, then
     auto-detect the config and model classes by inspecting the AST.

Usage (call before any AutoModelForCausalLM.from_pretrained):
    from scripts.register_model import register_qwen3_5_moe
    register_qwen3_5_moe(MODEL_ID, os.environ['HF_HOME'])
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path


def register_qwen3_5_moe(model_id: str, hf_home: str) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM

    slug = model_id.replace("/", "__")
    code_dir = Path(hf_home) / "model_code" / slug
    code_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch config.json ──────────────────────────────────────────────
    cfg_path = hf_hub_download(model_id, "config.json", local_dir=str(code_dir))
    with open(cfg_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "qwen3_5_moe")

    # Guard: already registered this session
    try:
        AutoConfig.for_model(model_type)
        print(f"[register_model] {model_type} already registered — skipping.")
        return
    except (KeyError, ValueError):
        pass

    # ── 2. Resolve .py files ──────────────────────────────────────────────
    auto_map = cfg.get("auto_map", {})

    if auto_map:
        # Fast path: auto_map tells us exactly which files/classes to use
        py_refs = {k: v for k, v in auto_map.items()
                   if k in ("AutoConfig", "AutoModelForCausalLM")}
        _download_auto_map(model_id, code_dir, py_refs)
    else:
        # Fallback: download all .py files from the repo and scan them
        print(f"[register_model] No auto_map found — downloading repo .py files ...")
        snapshot_download(
            model_id,
            local_dir=str(code_dir),
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf",
                             "*.ot", "*.msgpack", "flax_model*", "tf_model*"],
        )

    # ── 3. Find and register config + model classes ───────────────────────
    ConfigClass = _find_and_register_config(model_type, code_dir, auto_map, AutoConfig)
    if ConfigClass is None:
        raise RuntimeError(
            f"Could not find a PretrainedConfig subclass for {model_type!r} "
            f"in files downloaded from {model_id}."
        )

    _find_and_register_model(model_type, code_dir, auto_map,
                             ConfigClass, AutoModelForCausalLM)


# ── helpers ───────────────────────────────────────────────────────────────

def _import_py(py_file: Path, module_name: str):
    """Import a .py file as a module and return it."""
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _download_auto_map(model_id, code_dir, py_refs):
    from huggingface_hub import hf_hub_download
    for ref in py_refs.values():
        module_name = ref.rsplit(".", 1)[0]
        py_file = code_dir / f"{module_name}.py"
        if not py_file.exists():
            hf_hub_download(model_id, f"{module_name}.py",
                            local_dir=str(code_dir))


def _first_class_inheriting(py_file: Path, base_names: list[str]) -> str | None:
    """Return the first class name in py_file that inherits from any of base_names."""
    try:
        tree = ast.parse(py_file.read_text())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_str = ast.unparse(base)
            if any(b in base_str for b in base_names):
                return node.name
    return None


def _find_and_register_config(model_type, code_dir, auto_map, AutoConfig):
    if "AutoConfig" in auto_map:
        ref = auto_map["AutoConfig"]
        module_name, class_name = ref.rsplit(".", 1)
        mod = _import_py(code_dir / f"{module_name}.py", module_name)
        cls = getattr(mod, class_name)
    else:
        cls = _scan_for_class(
            code_dir,
            base_names=["PretrainedConfig"],
            name_hints=["Config"],
            exclude_names=["PretrainedConfig", "GenerationConfig"],
        )
        if cls is None:
            return None

    AutoConfig.register(model_type, cls, exist_ok=True)
    print(f"[register_model] Registered AutoConfig ({cls.__name__}) for {model_type}.")
    return cls


def _find_and_register_model(model_type, code_dir, auto_map,
                              ConfigClass, AutoModelForCausalLM):
    if "AutoModelForCausalLM" in auto_map:
        ref = auto_map["AutoModelForCausalLM"]
        module_name, class_name = ref.rsplit(".", 1)
        mod = _import_py(code_dir / f"{module_name}.py", module_name)
        cls = getattr(mod, class_name)
    else:
        cls = _scan_for_class(
            code_dir,
            base_names=["PreTrainedModel", "PretrainedModel"],
            name_hints=["ForCausalLM"],
            exclude_names=[],
        )
        if cls is None:
            print("[register_model] WARNING: could not find ForCausalLM class.")
            return

    AutoModelForCausalLM.register(ConfigClass, cls, exist_ok=True)
    print(f"[register_model] Registered AutoModelForCausalLM ({cls.__name__}) for {model_type}.")


def _scan_for_class(code_dir: Path, base_names: list[str],
                    name_hints: list[str], exclude_names: list[str]):
    """Scan all .py files in code_dir for a class matching hints."""
    candidates = []
    for py_file in sorted(code_dir.glob("*.py")):
        class_name = _first_class_inheriting(py_file, base_names)
        if class_name and class_name not in exclude_names:
            if any(h in class_name for h in name_hints):
                candidates.append((py_file, class_name))

    if not candidates:
        return None

    # Prefer files whose name contains 'modeling' or 'configuration'
    for hint in name_hints:
        for py_file, class_name in candidates:
            if hint.lower() in py_file.stem.lower():
                mod = _import_py(py_file, py_file.stem)
                return getattr(mod, class_name)

    py_file, class_name = candidates[0]
    mod = _import_py(py_file, py_file.stem)
    return getattr(mod, class_name)
