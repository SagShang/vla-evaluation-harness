# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "vla-eval",
#     "openpi",
#     "numpy>=1.24",
#     "pytest",
#     "chex",
# ]
#
# [tool.uv.sources]
# vla-eval = { path = "../../..", editable = true }
# openpi = { path = "../../../../openpi", editable = true }
#
# [tool.uv]
# exclude-newer = "2026-02-24T00:00:00Z"
# ///
"""π₀ / π₀.5 model server backed by the local OpenPI checkout."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import shutil

from vla_eval.model_servers.pi0 import Pi0ModelServer

logger = logging.getLogger(__name__)

_OPENPI_REPO = Path(__file__).resolve().parents[4] / "openpi"
_TRANSFORMERS_REPLACE = _OPENPI_REPO / "src/openpi/models_pytorch/transformers_replace"


def _is_pytorch_checkpoint(checkpoint: str | None) -> bool:
    return checkpoint is not None and (Path(checkpoint) / "model.safetensors").exists()


def _transformers_replace_installed() -> bool:
    importlib.invalidate_caches()
    try:
        check = importlib.import_module("transformers.models.siglip.check")
    except ImportError:
        return False
    return bool(check.check_whether_transformers_replace_is_installed_correctly())


def _ensure_transformers_replace() -> None:
    if not _TRANSFORMERS_REPLACE.is_dir():
        raise FileNotFoundError(f"OpenPI transformers_replace directory not found: {_TRANSFORMERS_REPLACE}")

    import transformers

    target_dir = Path(transformers.__file__).resolve().parent
    if not _transformers_replace_installed():
        logger.info("Patching transformers with OpenPI replacements: %s -> %s", _TRANSFORMERS_REPLACE, target_dir)
        shutil.copytree(_TRANSFORMERS_REPLACE, target_dir, dirs_exist_ok=True)
        importlib.invalidate_caches()
    if not _transformers_replace_installed():
        raise RuntimeError(f"Failed to install OpenPI transformers replacements into {target_dir}")


class Pi0LocalModelServer(Pi0ModelServer):
    """Reuse the standard π₀ server with the local OpenPI checkout."""

    def _load_model(self) -> None:
        if _is_pytorch_checkpoint(self.checkpoint):
            _ensure_transformers_replace()
        super()._load_model()


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(Pi0LocalModelServer)
