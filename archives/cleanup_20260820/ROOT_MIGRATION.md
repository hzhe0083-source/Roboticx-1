# Root migration — 2026-08-20

Keep `train.py`, `evaluate.py`, and `eval_metaworld.py` as compatibility entrypoints. Move standalone demos, benchmarks, secondary evaluators, data CLIs, migrations, and pure utility libraries into `scripts/` or `va_compound/`. Widely imported root modules require later extraction rather than mechanical moves. Historical paths inside `logs/` are never rewritten.
