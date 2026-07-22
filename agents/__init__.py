
def run_rca(state):
    from agents.rca import run_rca as _fn
    return _fn(state)


def run_maintenance(state):
    from agents.maintenance import run_maintenance as _fn
    return _fn(state)


def run_quality(state):
    from agents.quality import run_quality as _fn
    return _fn(state)


def run_regulation(state):
    from agents.regulation import run_regulation as _fn
    return _fn(state)


def run_failure_intelligence(state):
    from agents.failure import run_failure_intelligence as _fn
    return _fn(state)


__all__ = [
    "run_rca",
    "run_maintenance",
    "run_quality",
    "run_regulation",
    "run_failure_intelligence",
]
