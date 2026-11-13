"""Branching-only menu for the revised experiment; no sequential-chain actions."""

FIXED_TREE_CONFIG = (2, 2, 1)

BRANCHING_CONFIGS = (
    FIXED_TREE_CONFIG,
    (2, 1, 2), (1, 2, 2),
    (3, 1, 1), (1, 3, 1), (1, 1, 3),
    (3, 2, 1), (2, 3, 1), (1, 3, 2), (2, 1, 3),
    (2, 2, 2), (3, 1, 2),
    (2, 1, 1, 2), (1, 2, 1, 2), (2, 2, 1, 1),
    (4, 2, 1), (2, 4, 1), (1, 2, 4),
    (2, 3, 2), (3, 2, 2, 1), (2, 2, 3, 1),
    (4, 2, 2, 1), (1, 4, 2, 2), (2, 1, 2, 1, 2),
)


def validate_branching_configs(configs):
    if not configs or len(set(configs)) != len(configs):
        raise ValueError('Need distinct configurations')
    for config in configs:
        if not config or any(type(k) is not int or k < 1 for k in config):
            raise ValueError('Widths must be positive integers')
        if not any(k > 1 for k in config):
            raise ValueError(f'Pure chains are excluded: {config}')
    if FIXED_TREE_CONFIG not in configs:
        raise ValueError('The fixed baseline must be included in the action menu')
