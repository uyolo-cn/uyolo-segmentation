def naive_collate(batch):
    """Only collate dict value in to a list. E.g. meta data dict and img_info
    dict will be collated."""

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: naive_collate([d[key] for d in batch]) for key in elem}
    else:
        return batch