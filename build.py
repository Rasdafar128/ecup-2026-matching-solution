"""Сборка `bootstrap.ipynb` из дерева проекта.

На проверочную или арендованную машину едет не репозиторий, а одна тетрадь:
там есть Jupyter и больше ничего — ни git, ни scp, ни shell. Поэтому каждый
файл дерева кладётся в собственную `%%writefile`-ячейку, и **Run All**
воссоздаёт проект целиком.

    python build.py                  # пересобрать bootstrap.ipynb
    python build.py --out b.ipynb    # в другой файл
    python build.py --check          # ничего не писать, сверить с тем, что есть

Порядок работы: правим файлы в `src/`, гоняем `build.py`, отправляем четыре
тетради. Обратная правка — редактирование ячейки в самой тетради — молча
разойдётся с деревом, и следующая сборка её затрёт.

Зависимостей у скрипта нет: формат ipynb пишется напрямую, ставить `nbformat`
ради него не нужно. Данные и веса тетрадью не возят — данные качаются на месте
(`src/scripts/01_fetch_data.py`), веса обучаются заново.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent

ROOT_FILES = ("run.py", "requirements.txt")
TREE = "src"

SKIP_NAMES = {"build.py", "README.md", "LICENSE", ".DS_Store", "Thumbs.db"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", "venv",
             "data", "artifacts", "submissions", "build", "logs", "runs"}
SKIP_SUFFIXES = {".ipynb", ".pyc", ".pyo", ".parquet", ".zip", ".npz", ".npy",
                 ".safetensors", ".bin", ".csv", ".log"}

INTRO = """# E-CUP 2026 — bootstrap

**Run All** воссоздаёт дерево проекта в текущем каталоге и ставит зависимости.
Каждый файл едет своей `%%writefile`-ячейкой, поэтому тетрадь самодостаточна:
ни git, ни scp на машине не нужны — хватает одного Jupyter.

Перезапускать можно сколько угодно, файлы просто перезаписываются.

Данные и веса сюда не входят: данные качаются на месте
(`src/scripts/01_fetch_data.py`, хранилище организаторов), веса обучаются заново
тетрадями `01_mmbert.ipynb` и `02_bge.ipynb`.

Раздел 4 выравнивает оболочку с ядром Jupyter: `python3` и `pip` в `!`-командах
всех тетрадей — это тот же интерпретатор, в который ставятся зависимости.

Собрана `build.py` из дерева проекта; править файлы здесь бессмысленно —
следующая сборка затрёт правку.

После прогона этой тетради — по порядку:
`01_mmbert.ipynb` → `02_bge.ipynb` → `03_blend.ipynb`."""

INTERP_MD = """## 4. Интерпретатор

`!python3` и `!pip` в тетрадях обязаны попадать в тот же интерпретатор, что и
ядро Jupyter — иначе зависимости ставятся в один Python, а скрипты идут в другой.
Ячейка проверяет это и там, где команды нет или она смотрит не туда, кладёт
шим (`exec <ядро> "$@"`) в первый доступный на запись каталог из `PATH`:
`/usr/local/bin`, иначе `~/.local/bin`. Шимы — файлы, поэтому переживают
смену тетради; shell-алиасы в `!`-командах не живут."""

INTERP_CODE = """import os
import pathlib
import shutil
import subprocess
import sys

PY = sys.executable
KERNEL = pathlib.Path(PY).resolve()
print('ядро:', PY, sys.version.split()[0])


def points_to_kernel(name):
    exe = shutil.which(name)
    if exe is None:
        return None
    if name.startswith('python'):
        r = subprocess.run([exe, '-c', 'import sys; print(sys.executable)'],
                           capture_output=True, text=True)
        return r.returncode == 0 and pathlib.Path(r.stdout.strip()).resolve() == KERNEL
    r = subprocess.run([exe, '--version'], capture_output=True, text=True)
    return r.returncode == 0 and str(pathlib.Path(sys.prefix).resolve()) in r.stdout


if os.name != 'posix':
    print('не posix: шимы не ставятся, тетради зовут ядро как есть')
else:
    candidates = ['/usr/local/bin', os.path.expanduser('~/.local/bin')]
    bindir = next((pathlib.Path(d) for d in candidates
                   if pathlib.Path(d).is_dir() and os.access(d, os.W_OK)), None)
    if bindir is None:
        bindir = pathlib.Path(candidates[-1])
        bindir.mkdir(parents=True, exist_ok=True)
    if str(bindir) not in os.environ['PATH'].split(os.pathsep):
        os.environ['PATH'] = str(bindir) + os.pathsep + os.environ['PATH']

    shims = {'python': [PY], 'python3': [PY],
             'pip': [PY, '-m', 'pip'], 'pip3': [PY, '-m', 'pip']}
    for name, cmd in shims.items():
        state = points_to_kernel(name)
        if state:
            print(f'  {name:<8} уже ядро: {shutil.which(name)}')
            continue
        shim, tmp = bindir / name, bindir / f'.{name}.shim'
        tmp.write_text('#!/bin/sh\\nexec ' + ' '.join(f'"{c}"' for c in cmd) + ' "$@"\\n')
        tmp.chmod(0o755)
        os.replace(tmp, shim)
        why = 'не найден' if state is None else 'смотрел на другой интерпретатор'
        print(f'  {name:<8} {why} -> {shim}')

    for name in ('python3', 'pip'):
        if not points_to_kernel(name):
            raise SystemExit(f'{name} не совпадает с ядром даже после шима — '
                             f'проверьте PATH: ' + os.environ['PATH'])
    print('python3 и pip совпадают с ядром Jupyter, каталог шимов:', bindir)"""

DEPS_MD = """## 5. Зависимости

Ставятся через `python3 -m pip` — тем интерпретатором, который раздел 4 закрепил
за ядром. `torch` ставится с индекса CUDA 12.8 — на PyPI лежит другая сборка.
Остальное из `requirements.txt`, который только что записан выше."""

DEPS_CODE = """!python3 -m pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
!python3 -m pip install -q -r requirements.txt"""

ENV_CODE = """# Проверка окружения: версии и наличие карты.
import importlib

for m in ('torch', 'transformers', 'numpy', 'pyarrow', 'polars', 'sklearn'):
    try:
        mod = importlib.import_module(m)
        print(f'  {m:<14} {getattr(mod, "__version__", "?")}')
    except Exception as exc:
        print(f'  {m:<14} НЕ СТАВИТСЯ: {type(exc).__name__}: {exc}')

import torch
print('  cuda:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
print()
import src.ecup.solve, src.solution, src.utils.blend
print('дерево импортируется — можно запускать 01_mmbert.ipynb')"""


def skipped(p: Path) -> bool:
    return (p.name in SKIP_NAMES
            or p.suffix.lower() in SKIP_SUFFIXES
            or any(part in SKIP_DIRS for part in p.parts))


def collect(root: Path) -> list[Path]:
    files = [Path(n) for n in ROOT_FILES]
    missing = [f for f in files if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"нет обязательных файлов: {[str(m) for m in missing]}")
    tree = root / TREE
    if not tree.is_dir():
        raise SystemExit(f"нет каталога {TREE}/ — собирать нечего")
    files += sorted((p.relative_to(root) for p in tree.rglob("*")
                     if p.is_file() and not skipped(p.relative_to(root))),
                    key=lambda p: p.as_posix())
    return files


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return text if text.endswith("\n") else text + "\n"


def cell(kind: str, source: str, n: int) -> dict:
    c = {"cell_type": kind, "id": f"c{n:03d}", "metadata": {},
         "source": source.split("\n")}
    c["source"] = [s + "\n" for s in c["source"][:-1]] + c["source"][-1:]
    if not c["source"][-1]:
        c["source"].pop()
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


def notebook(root: Path, files: list[Path]) -> dict:
    dirs = sorted({f.parent.as_posix() for f in files if f.parent.as_posix() != "."})
    cells: list[dict] = []

    def add(kind: str, source: str) -> None:
        cells.append(cell(kind, source, len(cells)))

    add("markdown", INTRO)

    add("markdown", "## 1. Каталоги")
    mk = ["import os, pathlib, sys", ""]
    mk += [f"os.makedirs({d!r}, exist_ok=True)" for d in dirs]
    mk += [
        "",
        "ROOT = pathlib.Path.cwd().resolve()",
        "os.environ['ECUP_ROOT'] = str(ROOT)",
        "if str(ROOT) not in sys.path:",
        "    sys.path.insert(0, str(ROOT))",
        f"print('каталогов создано:', {len(dirs)}, '| ECUP_ROOT =', ROOT)",
    ]
    add("code", "\n".join(mk))

    add("markdown", f"## 2. Файлы проекта — {len(files)} штук")
    for f in files:
        add("code", f"%%writefile {f.as_posix()}\n{read_text(root / f)}")

    add("markdown", "## 3. Проверка, что дерево на месте")
    listing = ",\n".join(f'"{f.as_posix()}"' for f in files)
    add("code",
        "import os\n\n"
        f"expected = [\n{listing}\n]\n\n"
        "missing = [p for p in expected if not os.path.exists(p)]\n"
        "if missing:\n"
        "    raise SystemExit('не записались: ' + ', '.join(missing))\n"
        f"print('все {len(files)} файлов на месте')")

    add("markdown", INTERP_MD)
    add("code", INTERP_CODE)

    add("markdown", DEPS_MD)
    add("code", DEPS_CODE)
    add("code", ENV_CODE)

    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bootstrap.ipynb", help="куда писать тетрадь")
    ap.add_argument("--check", action="store_true",
                    help="не писать, а сверить: расходится ли тетрадь с деревом")
    args = ap.parse_args()

    files = collect(HERE)
    nb = notebook(HERE, files)
    text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    out = HERE / args.out

    if args.check:
        if not out.is_file():
            print(f"{args.out}: нет файла")
            return 1
        same = out.read_text(encoding="utf-8") == text
        print(f"{args.out}: {'совпадает с деревом' if same else 'РАСХОДИТСЯ с деревом'}")
        return 0 if same else 1

    out.write_text(text, encoding="utf-8", newline="\n")
    size = out.stat().st_size
    print(f"{args.out}: {len(files)} файлов, {len(nb['cells'])} ячеек, "
          f"{size / 1024:.0f} КБ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
