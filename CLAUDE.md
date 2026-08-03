# Development Guidelines

This codebase is a python package for live gameplay and analysis of a cardgame described in the README. It uses flask-socketio in the web application extras, and jinja templating for rich display in the standard application.

## Core Development Rules

1. Package Management
   - ONLY use uv to manage the package and installation
   - Running tools: `uv run tool`. Do not invoke python directly but use `uv run script.py`, use --with if the script has dependencies eg: `uv run --with numpy example.py`

4. Code Style
    - constants should be grouped near the top of any file and in UPPER_SNAKE_CASE
    - Where code concerns can be logically separated use different files
    - Avoid naming contributors, ensure that a github noreply email is used for commits
    - Keep code and comments brief - describe the functions purpose but do not write long comments on the implementation details

3. Evaluation improvement validation
   - When changes are made to `choose_move` or `Game.evaluate` tests should be run to ensure the output is identical to previous versions
   - A labelled corpus is generated on demand (not checked in - the older `data/*.jsonl` corpora were removed as their save strings are incompatible with the current `Game.load`): `python -m cardgame.validation.oracle generate --positions 800 --out data/oracle_labels.jsonl --jobs 8`
   - Changes to the heuristics in `choose_move` must be checked for percentage agreement on a sample of oracle positions, and using the tooling in `cardgame.validation.arena` to check for improvements against the previous version
   - Only a limited depth of search (normally 14 cards) should be used for validation runs as the time increases signficantly

4. Test results
   - When testing performance always append partial results to a file rather than storing in memory so the command can be stopped early if needed
   - If testing performance improvements to a search function like Game.evaluate or other expectiminimax search compare timings on at least 3 positions