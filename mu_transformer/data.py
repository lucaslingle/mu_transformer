# Copyright 2024 Lucas Dax Lingle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
import os
import posixpath
from typing import Optional

import blobfile
import datasets as hfds
import numpy as np
import tqdm
import transformers as hftr
from absl import logging


def get_tokenizer(
    tokenizer_name: str,
    model_name: Optional[str] = None,
    pad_token: Optional[str] = None,
) -> hftr.PreTrainedTokenizerFast:
    # get class
    cls = getattr(hftr, tokenizer_name)
    # instantiate class
    kwargs = dict(pad_token=pad_token) if pad_token is not None else dict()
    if model_name is not None:
        obj = cls.from_pretrained(model_name, **kwargs)
    else:
        try:
            model_name, *_ = tokenizer_name.lower().split("tokenizer")
            obj = cls.from_pretrained(model_name, **kwargs)
        except Exception as e:
            raise NotImplementedError(f"Got exception {e}.")
    # grab eos token, for consistency of data pipeline always use it for padding
    if pad_token is None:
        assert obj.eos_token_id is not None
        obj = get_tokenizer(tokenizer_name, model_name, pad_token=obj.eos_token)
    assert obj.is_fast
    return obj


def get_shard_fp(workdir, identifier, split_name, pcount, pindex):
    return posixpath.join(
        workdir, "data", identifier, f"{pcount}", f"{split_name}-{pindex}.bin"
    )


def get_arr_dtype(vocab_size):
    assert vocab_size < 65_000
    return np.uint16


# todo: maybe fix up the splitting logic to use the official vat/test split as test,
#   if only one of them is provided.
#   can then use a subset of official training split for validation,
#   if an official validation split is not available.
def write_dataset_to_memmap(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,  # batch size per host
    sequence_len: int,
    pcount: int,
    pindex: int,
    workdir: str,
) -> str:
    workdir_fp = get_shard_fp(workdir, hfds_identifier, split_name, pcount, pindex)
    temp_fp = posixpath.join("/tmp/", posixpath.split(workdir_fp)[-1])

    if blobfile.exists(workdir_fp):
        logging.info(f"Mem-mapped file exists at {workdir_fp}, skipping write...")
        return workdir_fp
    if os.path.exists(temp_fp):
        os.remove(temp_fp)

    # get tokenizer info
    assert hftr_tokenizer.is_fast

    # get available splits, and pick one.
    hfds_splits_set = set(hfds.get_dataset_split_names(hfds_identifier))
    if hfds_splits_set != {"train", "validation", "test"}:
        # we'll split the training data later, since there aren't enough provided splits
        hfds_split = "train"
    else:
        logging.info(f"hfds_splits_set: {hfds_splits_set}")
        hfds_split = split_name
    assert hfds_split in hfds_splits_set

    # load dataset lazily
    ds = hfds.load_dataset(
        hfds_identifier,
        hfds_config,
        split=hfds_split,
        streaming=True,
    )

    # shard by host, then tokenize the host's shard only
    def processing_fn(examples):
        examples = examples[hfds_datacol]
        examples = [e for i, e in enumerate(examples) if i % pcount == pindex]
        ids = hftr_tokenizer(
            examples,
            padding="max_length",
            truncation=True,
            max_length=sequence_len,
        )["input_ids"]
        return {"ids": ids}

    ds = ds.map(
        processing_fn,
        batched=True,
        batch_size=hfds_buffer_size * pcount,
        remove_columns=list(ds.column_names),
    )

    # whatever the official split we're working with happens to be,
    # need to shard by host and drop remainder
    dataset_info = list(hfds.get_dataset_infos(hfds_identifier).values())[0]
    try:
        canonical_count = dataset_info.splits.get(hfds_split).num_examples
    except AttributeError as exep:
        logging.error("You're using a bad dataset, it has no num_examples metadata...")
        raise exep
    sharded_canonical_count = canonical_count // pcount
    ds = ds.take(sharded_canonical_count)

    # if need be, split the training set into train/validation/test.
    # also, store the count for what's selected
    if hfds_splits_set != {"train", "validation", "test"}:
        sharded_val_count = batch_size * 100
        if split_name == "validation":
            sharded_split_count = sharded_val_count
            ds = ds.take(sharded_split_count)
        elif split_name == "test":
            sharded_split_count = sharded_val_count
            ds = ds.skip(sharded_val_count).take(sharded_split_count)
        elif split_name == "train":
            sharded_split_count = sharded_canonical_count - 2 * sharded_val_count
            ds = ds.skip(2 * sharded_val_count).take(sharded_split_count)
        else:
            raise NotImplementedError("Unrecognized split name")
    else:
        sharded_split_count = sharded_canonical_count
        ds = ds.take(sharded_split_count)

    # note that currently the shards on all hosts have the same example count.
    # in addition, we want this example count to be divisible by the batch size per host
    # and by the write buffer size.
    write_buffer_size = hfds_buffer_size
    lcm = math.lcm(write_buffer_size, batch_size)
    writable_count = (sharded_split_count // lcm) * lcm
    assert writable_count > 0
    assert writable_count % batch_size == 0
    assert writable_count % write_buffer_size == 0

    # so make an iterator
    ds = ds.take(writable_count)
    ds = ds.iter(batch_size=write_buffer_size, drop_last_batch=True)

    # write to memmapped file
    n_shard_tokens = writable_count * sequence_len
    n_write_tokens_per_iter = write_buffer_size * sequence_len
    n_write_iters = writable_count // write_buffer_size
    logging.info(f"n_shard_tokens: {n_shard_tokens}")
    logging.info(f"n_write_tokens_per_iter: {n_write_tokens_per_iter}")
    logging.info(f"n_write_iters: {n_write_iters}")
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(temp_fp, dtype=arr_dtype, mode="w+", shape=(n_shard_tokens,))
    idx = 0
    for _ in tqdm.tqdm(range(n_write_iters), desc=f"Writing {temp_fp} with memmap"):
        batch = None
        while batch is None:
            try:
                batch = next(ds)["ids"]
            except BaseException as e:
                pass
        arr_batch = np.array(batch, dtype=arr_dtype).reshape(-1)
        arr[idx : idx + n_write_tokens_per_iter] = arr_batch
        idx += n_write_tokens_per_iter
    arr.flush()

    logging.info(f"Copying {temp_fp} to {workdir_fp}")
    blobfile.copy(temp_fp, workdir_fp, overwrite=True)
    return workdir_fp


def read_dataset_to_memmap(
    hfds_identifier: str,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    pcount: int,
    pindex: int,
    workdir: str,
    force_download: bool,
) -> np.ndarray:
    workdir_fp = get_shard_fp(workdir, hfds_identifier, split_name, pcount, pindex)
    temp_fp = posixpath.join("/tmp/", posixpath.split(workdir_fp)[-1])

    if force_download or not blobfile.exists(temp_fp):
        logging.info(f"Copying {workdir_fp} to {temp_fp}")
        blobfile.copy(workdir_fp, temp_fp, overwrite=True)

    logging.info(f"Reading with np.memmap...")
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(temp_fp, dtype=arr_dtype, mode="r")
    return arr


def get_dataset(
    hfds_identifier: str,
    hfds_config: str,
    hfds_datacol: str,
    hfds_buffer_size: int,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    batch_size: int,  # batch size per host
    sequence_len: int,
    pcount: int,
    pindex: int,
    workdir: str,
    force_download: bool,
) -> np.ndarray:
    logging.info("Calling write_dataset_to_memmap...")
    _ = write_dataset_to_memmap(
        hfds_identifier=hfds_identifier,
        hfds_config=hfds_config,
        hfds_datacol=hfds_datacol,
        hfds_buffer_size=hfds_buffer_size,
        hftr_tokenizer=hftr_tokenizer,
        split_name=split_name,
        batch_size=batch_size,
        sequence_len=sequence_len,
        pcount=pcount,
        pindex=pindex,
        workdir=workdir,
    )
    logging.info("Calling read_dataset_to_memmap...")
    arr = read_dataset_to_memmap(
        hfds_identifier=hfds_identifier,
        hftr_tokenizer=hftr_tokenizer,
        split_name=split_name,
        pcount=pcount,
        pindex=pindex,
        workdir=workdir,
        force_download=force_download,
    )
    return arr


def get_batch(arr, batch_size, sequence_len, step, out_dtype=np.int32):
    # todo: support shuffle indices
    chunk_size = batch_size * sequence_len
    n_chunks = arr.shape[0] // chunk_size
    assert arr.shape[0] == n_chunks * chunk_size
    folded_step = step % n_chunks
    batch = arr[chunk_size * folded_step : chunk_size * (folded_step + 1)]
    batch = np.reshape(batch, [batch_size, sequence_len])
    batch = batch.astype(out_dtype)
    return batch


def count_batches(arr, batch_size, sequence_len):
    assert arr.ndim == 1
    count = arr.shape[0]
    assert count % (batch_size * sequence_len) == 0
    return count // (batch_size * sequence_len)
