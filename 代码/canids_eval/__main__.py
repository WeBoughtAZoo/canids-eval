# -*- coding: utf-8 -*-
"""`python -m canids_eval ...` 的入口（等价于 `python -m canids_eval.cli ...`）。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
