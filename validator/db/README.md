# Validator Database

This package contains validator persistence code.

- `database.py` wraps PostgreSQL connection pooling.
- `constants.py` names tables and columns used by SQL helpers.
- `migrations/` is the DB compatibility contract and should remain append-only except for deliberate migration fixes.
- `sql/` contains query helpers grouped by domain.

API and DB compatibility are the main backwards-compatibility boundary for this refactor.
