def chronological_boundaries(n_times: int, train_fraction: float, val_fraction: float):
    train_end = int(n_times * train_fraction)
    val_end = int(n_times * (train_fraction + val_fraction))
    if not (0 < train_end < val_end < n_times):
        raise ValueError("Invalid chronological split fractions")
    return train_end, val_end
