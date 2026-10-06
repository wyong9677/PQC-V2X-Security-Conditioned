#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PATCH_LEVEL = "E4_R1_NATIVE_REBENCH_LINKER_FIX"


def dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if not x:
            continue
        p = str(Path(x).expanduser())
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def run_capture(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def brew_prefix(formula: str) -> str | None:
    brew = shutil.which("brew")
    if not brew:
        return None
    rc, out, _ = run_capture([brew, "--prefix", formula])
    if rc == 0 and out:
        return out.splitlines()[-1].strip()
    return None


def candidate_prefixes() -> list[str]:
    home = Path.home()
    vals = [
        os.environ.get("OQS_ROOT", ""),
        os.environ.get("LIBOQS_ROOT", ""),
        brew_prefix("liboqs") or "",
        str(home / "ResearchTools/openssl_oqs/liboqs-install"),
        "/opt/homebrew/opt/liboqs",
        "/opt/homebrew",
        "/usr/local",
    ]
    return dedupe(vals)


def pkg_config_flags(extra_pc_dirs: list[str] | None = None, static: bool = False):
    pc = shutil.which("pkg-config")
    if not pc:
        return None
    env = os.environ.copy()
    if extra_pc_dirs:
        existing = env.get("PKG_CONFIG_PATH", "")
        env["PKG_CONFIG_PATH"] = os.pathsep.join(extra_pc_dirs + ([existing] if existing else []))
    rc, _, _ = run_capture([pc, "--exists", "liboqs"], env=env)
    if rc != 0:
        return None
    rc1, cflags, e1 = run_capture([pc, "--cflags", "liboqs"], env=env)
    lib_cmd = [pc]
    if static:
        lib_cmd.append("--static")
    lib_cmd += ["--libs", "liboqs"]
    rc2, libs, e2 = run_capture(lib_cmd, env=env)
    if rc1 or rc2:
        return None
    return {
        "cflags": shlex.split(cflags),
        "libs": shlex.split(libs),
        "stderr": "\n".join(x for x in (e1, e2) if x),
        "env": env,
    }


def openssl_prefixes() -> list[str]:
    home = Path.home()
    vals = [
        os.environ.get("OPENSSL_ROOT_DIR", ""),
        os.environ.get("OPENSSL_ROOT", ""),
        brew_prefix("openssl@3") or "",
        str(home / "ResearchTools/openssl_oqs/openssl-install"),
        str(home / "ResearchTools/openssl_oqs/openssl-install-3"),
        "/opt/homebrew/opt/openssl@3",
        "/usr/local/opt/openssl@3",
    ]
    return dedupe(vals)


def explicit_recipes(prefix: str) -> list[dict]:
    p = Path(prefix)
    header = p / "include/oqs/oqs.h"
    if not header.exists():
        return []
    libdirs = [d for d in (p / "lib", p / "lib64") if d.exists()]
    recipes: list[dict] = []

    pc_dirs = [str(d / "pkgconfig") for d in libdirs if (d / "pkgconfig").exists()]
    if pc_dirs:
        for static in (False, True):
            flags = pkg_config_flags(pc_dirs, static=static)
            if flags:
                recipes.append({
                    "method": f"prefix-pkg-config{'-static' if static else ''}",
                    "prefix": str(p),
                    "cflags": flags["cflags"],
                    "libs": flags["libs"],
                    "env": flags["env"],
                })

    for libdir in libdirs:
        dylibs = [libdir / "liboqs.dylib", libdir / "liboqs.so"]
        for lib in dylibs:
            if lib.exists():
                recipes.append({
                    "method": "explicit-shared-library",
                    "prefix": str(p),
                    "cflags": [f"-I{p / 'include'}"],
                    "libs": [str(lib), f"-Wl,-rpath,{libdir}", "-lm"],
                    "env": os.environ.copy(),
                })

        static_lib = libdir / "liboqs.a"
        if static_lib.exists():
            base = [str(static_lib)]
            recipes.append({
                "method": "explicit-static-basic",
                "prefix": str(p),
                "cflags": [f"-I{p / 'include'}"],
                "libs": base + ["-lm"],
                "env": os.environ.copy(),
            })
            for op in openssl_prefixes():
                opath = Path(op)
                if (opath / "include/openssl/crypto.h").exists() and (opath / "lib").exists():
                    recipes.append({
                        "method": "explicit-static-openssl3",
                        "prefix": str(p),
                        "cflags": [f"-I{p / 'include'}", f"-I{opath / 'include'}"],
                        "libs": base + [f"-L{opath / 'lib'}", "-lcrypto", "-lm"],
                        "env": os.environ.copy(),
                    })
                    break
    return recipes


def all_recipes() -> list[dict]:
    recipes: list[dict] = []
    global_flags = pkg_config_flags(static=False)
    if global_flags:
        recipes.append({
            "method": "pkg-config",
            "prefix": "",
            "cflags": global_flags["cflags"],
            "libs": global_flags["libs"],
            "env": global_flags["env"],
        })
    global_static = pkg_config_flags(static=True)
    if global_static:
        recipes.append({
            "method": "pkg-config-static",
            "prefix": "",
            "cflags": global_static["cflags"],
            "libs": global_static["libs"],
            "env": global_static["env"],
        })
    for prefix in candidate_prefixes():
        recipes.extend(explicit_recipes(prefix))

    # Deduplicate exact command-flag recipes.
    out = []
    seen = set()
    for r in recipes:
        key = (tuple(r["cflags"]), tuple(r["libs"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def self_test() -> int:
    assert shlex.split("-L/tmp/oqs/lib -loqs -lm") == ["-L/tmp/oqs/lib", "-loqs", "-lm"]
    assert dedupe(["/a", "/a", "/b"]) == ["/a", "/b"]
    print("E4R1_BUILD_HELPER_SELF_TEST=PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--output")
    ap.add_argument("--status-json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.source or not args.output or not args.status_json:
        ap.error("--source, --output and --status-json are required")

    src = Path(args.source)
    out = Path(args.output)
    status_path = Path(args.status_json)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    clang = "/usr/bin/clang" if Path("/usr/bin/clang").exists() else (shutil.which("clang") or "clang")

    attempts = []
    recipes = all_recipes()
    if not recipes:
        status = {
            "patch_level": PATCH_LEVEL,
            "native_liboqs_found": False,
            "build_ok": False,
            "reason": "NO_USABLE_LIBOQS_LINK_RECIPE_FOUND",
            "candidate_prefixes": candidate_prefixes(),
            "attempts": [],
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print("E4R1_NATIVE_LIBOQS_FOUND=NO")
        print("E4R1_NATIVE_BENCH_BUILD=NOT_ATTEMPTED")
        return 10

    print(f"E4R1_NATIVE_LINK_RECIPES={len(recipes)}")
    for idx, r in enumerate(recipes, 1):
        cmd = [clang, "-O3", "-std=c11", *r["cflags"], str(src), *r["libs"], "-o", str(out)]
        print(f"E4R1_LINK_ATTEMPT={idx} method={r['method']} prefix={r['prefix'] or 'pkg-config-default'}")
        print("E4R1_LINK_ARGV=" + json.dumps(cmd))
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=r.get("env"))
        attempts.append({
            "index": idx,
            "method": r["method"],
            "prefix": r["prefix"],
            "cflags": r["cflags"],
            "libs": r["libs"],
            "returncode": p.returncode,
            "stdout_tail": p.stdout[-4000:],
            "stderr_tail": p.stderr[-8000:],
        })
        if p.returncode == 0 and out.exists():
            status = {
                "patch_level": PATCH_LEVEL,
                "native_liboqs_found": True,
                "build_ok": True,
                "selected_method": r["method"],
                "selected_prefix": r["prefix"],
                "compiler": clang,
                "output": str(out),
                "attempts": attempts,
            }
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            print("E4R1_NATIVE_LIBOQS_FOUND=YES")
            print(f"E4R1_NATIVE_LINK_METHOD={r['method']}")
            print(f"E4R1_NATIVE_LINK_PREFIX={r['prefix'] or 'pkg-config-default'}")
            print("E4R1_NATIVE_BENCH_BUILD=PASS")
            return 0
        try:
            out.unlink()
        except FileNotFoundError:
            pass

    status = {
        "patch_level": PATCH_LEVEL,
        "native_liboqs_found": True,
        "build_ok": False,
        "reason": "ALL_LINK_RECIPES_FAILED",
        "candidate_prefixes": candidate_prefixes(),
        "attempts": attempts,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print("E4R1_NATIVE_LIBOQS_FOUND=YES")
    print("E4R1_NATIVE_BENCH_BUILD=FAIL")
    for a in attempts[-3:]:
        if a["stderr_tail"]:
            print(f"E4R1_LINK_FAILURE_TAIL[{a['index']}]=")
            print(a["stderr_tail"])
    return 11


if __name__ == "__main__":
    raise SystemExit(main())
