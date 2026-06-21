"""Marks ``tests`` as a package so cross-test imports (e.g. shared fake providers in
``test_model_pool``) resolve under a bare ``pytest`` invocation, not only ``python -m pytest``
(which happens to put the cwd on sys.path). Keeps CI — which runs bare ``pytest`` — green.
"""
