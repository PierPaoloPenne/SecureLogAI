# Team Contributions

Template for tracking individual contributions across the project — intended
to back up each member's individual reflective journal with a concrete,
commit-referenced record rather than a vague summary written from memory at
the end. Fill in names, dates, and commit/PR links as the project
progresses; update it at the end of each week rather than all at once at
the deadline.

## Team

| Name | Role focus | GitHub handle |
|---|---|---|
| _Team Member 1_ | NLP / ML pipeline (parser, features, model) | `@` |
| _Team Member 2_ | Web interface, integration, security hardening | `@` |

> Roles overlapped in practice — planning, testing, and documentation were
> shared. This table reflects primary ownership, not the only work each
> person did.

## Contribution log

| Week | Member | Contribution | Files / commits |
|---|---|---|---|
| 1 | | Problem definition, market research, architecture sketch | |
| 1 | | Repo scaffolding, requirements.txt, .gitignore | |
| 2 | | `app/parser.py` — log format detection, syslog/Apache/CSV parsing | |
| 2 | | `tests/test_parser.py` | |
| 3 | | `app/features.py` — temporal/severity/volume/text features | |
| 3 | | `notebooks/anomaly_experiments.ipynb` — feature & model exploration | |
| 4 | | `app/model.py` — Isolation Forest wrapper, scoring | |
| 4 | | `tests/test_model.py` | |
| 4-5 | | `app/main.py`, `app/templates/` — FastAPI app + UI | |
| 5 | | Upload validation & security hardening (`app/utils.py`) | |
| 6 | | `README.md`, `docs/architecture.md` | |
| 6 | | Demo prep, final testing pass | |

## Commit summary

_Populate before submission, e.g. via:_
```bash
git shortlog -sne
```
_which lists each contributor's commit count — useful as a sanity check
against the table above, not a replacement for it (commit count alone
doesn't reflect design/debugging effort)._

## Individual reflective journal pointers

Each member's reflective journal (max 1000 words, per the assignment brief)
should reference specific rows from the table above rather than restating
the whole project. Suggested structure per journal:

1. **Personal reflection** — what you found challenging, what you'd do
   differently, one specific decision you own.
2. **Technical reflection** — a concrete technical choice you made (e.g.
   "why I chose TF-IDF + SVD over a full embedding model", or "why the
   upload size cap is set where it is") and the trade-off behind it.
3. **Limitations** — what your component doesn't handle well, tied to a
   specific limitation in the README's Limitations & Scope section.
4. **Scope of contribution** — reference the specific files/weeks from the
   table above that were primarily yours.
