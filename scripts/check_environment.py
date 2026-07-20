"""Environment validation script for the Industrial PDF-to-Graph pipeline.

Run this before starting the main service to fail fast on critical mismatches
and provide clear remediation steps (Neo4j password expiration, Torch ABI, transformers).
"""
from __future__ import annotations

import os
import sys
import traceback
from importlib import import_module
from importlib.metadata import version, PackageNotFoundError

from app.pipeline.compat import ensure_pyarrow_compat


def _is_kaggle_environment() -> bool:
    return os.environ.get("KAGGLE_ENV") == "kaggle" or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "1"


def _safe_version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def check_neo4j_env() -> bool:
    uri = os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_BOLT_URI")
    user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER") or "neo4j"
    pwd = os.environ.get("NEO4J_PASSWORD")

    if not uri or not pwd:
        print("[env-check] Neo4j: URI or password not set; skipping active check")
        return True

    try:
        neo4j = import_module("neo4j")
    except Exception:
        print("[env-check] Neo4j driver not installed; cannot perform live check")
        return True

    try:
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, pwd), max_connection_lifetime=30)
        with driver.session() as session:
            res = session.run("RETURN 1 AS ok")
            _ = list(res)
        driver.close()
        print("[env-check] Neo4j: connection OK")
        return True
    except Exception as exc:
        msg = str(exc)
        print(f"[env-check] Neo4j connection failed: {type(exc).__name__} - {msg}")
        if "CredentialsExpired" in msg or "password change" in msg.lower() or "must change" in msg.lower():
            print("[env-check][action] Neo4j requires password change. Run:")
            print("  cypher-shell -u neo4j -p neo4j")
            print("  ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'YourNewPassword!';")
            print("  Then update .env.local NEO4J_PASSWORD and restart the service.")
        return False


def check_torch_abi() -> bool:
    ensure_pyarrow_compat()
    try:
        import torch
    except Exception:
        print("[env-check] torch not installed; skipping Torch ABI checks")
        return True

    try:
        import torchvision
    except Exception:
        print("[env-check] torchvision not installed; skip matching check")
        return True

    try:
        import torchaudio
    except Exception:
        print("[env-check] torchaudio not installed; skip matching check")
        return True

    tv = getattr(torch, "__version__", "?")
    vv = getattr(torchvision, "__version__", "?")
    av = getattr(torchaudio, "__version__", "?")

    print(f"[env-check] torch={tv}, torchvision={vv}, torchaudio={av}")

    # Compare major.minor
    def major_minor(v: str) -> str:
        parts = v.split(".")
        return ".".join(parts[:2]) if parts and parts[0] != "?" else v

    if _is_kaggle_environment():
        if major_minor(tv) != major_minor(av):
            print("[env-check][warn] Torch and torchaudio versions differ on Kaggle; this may be acceptable if Kaggle installed a compatible runtime")
        if major_minor(tv) != major_minor(vv):
            print("[env-check][warn] Torch and torchvision versions differ on Kaggle; this may be acceptable if Kaggle installed a compatible runtime")
        return True

    if major_minor(tv) != major_minor(av):
        print("[env-check][error] Torch and torchaudio ABI mismatch — reinstall matching versions")
        return False

    if major_minor(tv) != major_minor(vv):
        print("[env-check][error] Torch and torchvision ABI mismatch — reinstall matching versions")
        return False

    return True


def check_transformers_compat() -> bool:
    tr = _safe_version("transformers")
    if not tr:
        print("[env-check] transformers not installed; many NLP models may fail")
        return True

    print(f"[env-check] transformers=={tr}")
    # Try a minimal import pattern used by downstream libs
    try:
        mod = import_module("transformers.utils.import_utils")
        if not hasattr(mod, "is_opentelemetry_available"):
            print("[env-check][warn] transformers missing 'is_opentelemetry_available' helper — some packages may expect a different transformers version")
            # Not fatal by itself
        return True
    except Exception:
        print("[env-check][error] transformers import check failed")
        traceback.print_exc()
        return False


def check_pyarrow_version() -> bool:
    pa = _safe_version("pyarrow")
    if not pa:
        print("[env-check] pyarrow not installed; many data-processing libraries may fail")
        return False

    print(f"[env-check] pyarrow=={pa}")
    try:
        major, minor = map(int, pa.split(".")[:2])
    except Exception:
        return True

    if major > 15:
        print("[env-check][error] pyarrow version is too new for the pinned ML stack. Use pyarrow==15.0.2")
        return False

    return True


def check_docling_import() -> bool:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        print("[env-check] docling import OK")
        return True
    except Exception as exc:
        print(f"[env-check][error] docling import failed: {type(exc).__name__} - {exc}")
        return False


def check_sentence_transformers_import() -> bool:
    try:
        import sentence_transformers  # type: ignore
        print("[env-check] sentence-transformers import OK")
        return True
    except Exception as exc:
        print(f"[env-check][error] sentence-transformers import failed: {type(exc).__name__} - {exc}")
        return False


def check_gliner_import() -> bool:
    try:
        import gliner  # type: ignore
        print("[env-check] gliner import OK")
        return True
    except Exception as exc:
        print(f"[env-check][error] gliner import failed: {type(exc).__name__} - {exc}")
        return False


def check_glirel_import() -> bool:
    try:
        import glirel  # type: ignore
        print("[env-check] glirel import OK")
        return True
    except Exception as exc:
        print(f"[env-check][warn] glirel import failed: {type(exc).__name__} - {exc}")
        return True


def run_all_checks() -> bool:
    ok = True
    print("[env-check] Running environment validation checks...")
    try:
        if not check_neo4j_env():
            ok = False
        if not check_torch_abi():
            ok = False
        if not check_pyarrow_version():
            ok = False
        if not check_transformers_compat():
            ok = False
        if not check_docling_import():
            ok = False
        if not check_sentence_transformers_import():
            ok = False
        if not check_gliner_import():
            ok = False
        if not check_glirel_import():
            ok = False
    except Exception as exc:
        print(f"[env-check] Unexpected error during checks: {exc}")
        traceback.print_exc()
        ok = False

    if ok:
        print("[env-check] All critical checks passed (or skipped).")
    else:
        print("[env-check] One or more critical checks failed. See messages above and fix before starting the service.")

    return ok


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 2)
