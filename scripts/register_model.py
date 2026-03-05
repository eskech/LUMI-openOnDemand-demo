"""
Register qwen3_5_moe with transformers 4.52.4 at runtime.

qwen3_5_moe was added to transformers after 4.52.4, and the model's
config.json has no auto_map, so trust_remote_code=True cannot help.

This script downloads the model's Python files from the Hub (config +
modeling) and registers the classes with AutoConfig / AutoModelForCausalLM
so that from_pretrained works without upgrading transformers.

Usage (call before any AutoModelForCausalLM.from_pretrained):
    from scripts.register_model import register_qwen3_5_moe
    register_qwen3_5_moe(MODEL_ID, os.environ['HF_HOME'])
"""

import importlib.util
import json
import sys
from pathlib import Path


def register_qwen3_5_moe(model_id: str, hf_home: str) -> None:
    """Download model code from Hub and register with transformers Auto classes.

    Files are cached in <hf_home>/model_code/<model_slug>/ and reused on
    subsequent calls, so the download only happens once.
    """
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    slug = model_id.replace("/", "__")
    code_dir = Path(hf_home) / "model_code" / slug
    code_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Fetch config.json ──────────────────────────────────────────────
    cfg_path = hf_hub_download(model_id, "config.json", local_dir=str(code_dir))
    with open(cfg_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "qwen3_5_moe")

    # Already registered (e.g. this function was called earlier in the session)
    try:
        AutoConfig.for_model(model_type)
        print(f"[register_model] {model_type} already registered — skipping.")
        return
    except (KeyError, ValueError):
        pass

    # ── 2. Resolve class references ───────────────────────────────────────
    auto_map = cfg.get("auto_map", {})
    if not auto_map:
        raise RuntimeError(
            f"{model_id}: config.json has no auto_map and {model_type!r} is "
            "not in this version of transformers.\n"
            "Either upgrade transformers or supply model code manually."
        )

    def _load_class(ref: str) -> type:
        """Download <module>.py from Hub, import it, return the named class."""
        module_name, class_name = ref.rsplit(".", 1)
        py_file = code_dir / f"{module_name}.py"
        if not py_file.exists():
            hf_hub_download(model_id, f"{module_name}.py", local_dir=str(code_dir))
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)

    # ── 3. Register AutoConfig ────────────────────────────────────────────
    ConfigClass = None
    if "AutoConfig" in auto_map:
        ConfigClass = _load_class(auto_map["AutoConfig"])
        AutoConfig.register(model_type, ConfigClass, exist_ok=True)
        print(f"[register_model] Registered AutoConfig for {model_type}.")

    # ── 4. Register AutoModelForCausalLM ─────────────────────────────────
    if "AutoModelForCausalLM" in auto_map and ConfigClass is not None:
        ModelClass = _load_class(auto_map["AutoModelForCausalLM"])
        AutoModelForCausalLM.register(ConfigClass, ModelClass, exist_ok=True)
        print(f"[register_model] Registered AutoModelForCausalLM for {model_type}.")
