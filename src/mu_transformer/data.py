from typing import Iterator
from typing import Mapping, Optional

import datasets as hfds
import jax
import numpy as np
import transformers as hftr


def get_tokenizer(
    cls_name: str,
    short_name: Optional[str] = None,
    pad_token: Optional[str] = None,
) -> hftr.PreTrainedTokenizerFast:
    # get class
    cls = getattr(hftr, cls_name)
    # instantiate class
    kwargs = dict(pad_token=pad_token) if pad_token is not None else dict()
    if short_name is not None:
        obj = cls.from_pretrained(short_name, **kwargs)
    else:
        try:
            short_name, *_ = cls_name.lower().split("tokenizer")
            obj = cls.from_pretrained(short_name, **kwargs)
        except Exception as e:
            raise NotImplementedError(f"Got exception {e}.")
    # grab eos token, for consistency of data pipeline always use it for padding
    if pad_token is None:
        assert obj.eos_token_id is not None
        obj = get_tokenizer(cls_name, short_name, pad_token=obj.eos_token)
    assert obj.is_fast
    return obj


def get_dataset(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,
    sequence_len: int,
    start_step: int,
    shuffle: bool,
    shuffle_buffer_size: int = 1024,
) -> Iterator[Mapping[str, np.ndarray]]:

    # get shard info
    pcount = jax.process_count()
    pindex = jax.process_index()

    # get tokenizer info
    assert hftr_tokenizer.is_fast
    eos_id = hftr_tokenizer.eos_token_id
    bos_id = hftr_tokenizer.bos_token_id

    # get available splits, and pick one.
    hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier))
    if len(hfds_splits_set) == 1:
        # if only one split is available, we'll do this split ourselves later.
        hfds_split = set(y for y in hfds_splits_set).pop()
    elif split_name in hfds_splits_set:
        # use user-provided split name if possible
        hfds_split = split_name
    elif any(y.startswith(split_name) for y in hfds_splits_set):
        # use split that starts with the user-provided one
        hfds_split = set(y for y in hfds_splits_set if y.startswith(split_name)).pop()
    else:
        raise ValueError("Unrecognized split name.")

    # load dataset lazily
    ds = hfds.load_dataset(
        hfds_identifier,
        hfds_config,
        split=hfds_split,
        streaming=True,
    )

    # shard by host, then tokenize the host's shard only
    assert "content_" not in set(ds.column_names)

    def shard_by_host(examples):
        examples = examples[hfds_datacol]
        examples = [e for i, e in enumerate(examples) if i % pcount == pindex]
        return {"content_": examples}

    def tokenize(examples):
        targets = hftr_tokenizer(
            examples["content_"],
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )["input_ids"]
        inputs = [[bos_id, *e[0:-1]] for e in targets]
        return {"inputs": inputs, "targets": targets}

    ds = ds.map(
        shard_by_host,
        batched=True,
        batch_size=1024 * jax.process_count(),
        remove_columns=list(ds.column_names),
    )
    ds = ds.map(
        tokenize,
        batched=True,
        batch_size=1024,
    )

    # automatically split the training split if there aren't any other provided splits
    if len(hfds_splits_set) == 1:
        if split_name == "val":
            ds = ds.take(batch_size * 100)
        elif split_name == "test":
            ds = ds.take(batch_size * 200).skip(batch_size * 100)
        elif split_name == "train":
            ds = ds.skip(batch_size * 200)
        else:
            raise NotImplementedError("Unrecognized split name")

    # shuffle the training split
    if shuffle:
        assert split_name == "train"
        shuffle_seed = start_step + (10 ** 9) * pindex
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_size)

    # convert to iterator, batch examples to the desired batch size per host.
    ds_iter = ds.iter(batch_size=batch_size, drop_last_batch=True)
    ds_iter = map(
        lambda r: {
            "inputs": np.array(r["inputs"], dtype=np.int32),
            "targets": np.array(r["targets"], dtype=np.int32),
            "loss_mask": np.cumprod(
                np.pad(
                    np.not_equal(r["inputs"], eos_id)[:, 1:],
                    pad_width=((0, 0), (1, 0)),
                    constant_values=True,
                ).astype(np.int32),
                axis=-1,
            ),  # mask out every timestep where the input is eos, except at seq start
        },
        ds_iter,
    )
    return ds_iter
