# SPDX-FileCopyrightText: 2026 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
"""A module-level function in the report module must not reference `self`. report_worker runs in the report
worker process as a plain function over validator_data; a `self` that slipped into it (1 October 2026, the
newcomer ramp factor column) raised NameError on every cycle, the worker logged "Error in report worker" and
published nothing, and no unit test saw it because the per-agent dict is only built inside the worker."""
import ast
import inspect


def _self_in_plain_functions(module):
    tree = ast.parse(inspect.getsource(module))
    offenders = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = {a.arg for a in node.args.args + node.args.kwonlyargs}
            if "self" in params:
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id == "self":
                    offenders.append((node.name, sub.lineno))
    return offenders


def test_report_and_reward_plain_functions_never_use_self():
    from taos.im.validator import report, reward

    for mod in (report, reward):
        assert _self_in_plain_functions(mod) == [], mod.__name__
