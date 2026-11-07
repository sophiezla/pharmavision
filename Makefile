# Thin wrapper: every target is implemented in scripts/tasks.py so it also runs where make is absent.
PY ?= python
EXP ?=

.PHONY: setup lint test smoke aggregate figures freeze reproduce

setup lint test smoke freeze reproduce:
	$(PY) scripts/tasks.py $@

aggregate figures:
	$(PY) scripts/tasks.py $@ --exp $(EXP)
