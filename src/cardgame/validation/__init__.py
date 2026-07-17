"""Bot validation tooling: strength matches, oracle labeling/agreement,
and leaf-weight fitting.

* `arena` — duplicate-deal strength matches between two bot versions
  (`python -m cardgame.validation.arena --old HEAD --new current ...`)
* `oracle` — exact-solver labeling of late positions and fixed-depth
  move-agreement testing (`python -m cardgame.validation.oracle ...`)
* `fit_weights` — least-squares fit of the leaf evaluation's weights
  against oracle labels (`python -m cardgame.validation.fit_weights`)

Import the submodules directly (`from cardgame.validation.arena import
run_match`); the package deliberately re-exports nothing so `python -m`
invocations stay free of double-import warnings.
"""
