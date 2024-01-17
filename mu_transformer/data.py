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
import os.path as osp
import posixpath
from typing import Optional

import datasets as hfds
import gcsfs
import jax
import numpy as np
import tqdm
import transformers as hftr
from absl import logging


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


def get_shard_fname(workdir, identifier, split_name, pindex):
    return posixpath.join(workdir, "data", identifier, f"{split_name}-{pindex}.bin")


def get_arr_dtype(vocab_size):
    assert vocab_size < 65_000
    return np.uint16


# todo: maybe fix up the splitting logic to use the official vat/test split as test,
#   if only one of them is provided.
#   can then use a subset of official training split for validation,
#   if an official validation split is not available.
def write_dataset_to_memmmap(
    gc_project: str,
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
    cloud_fp = get_shard_fname(workdir, hfds_identifier, split_name, pindex)
    cloud_fs = gcsfs.GCSFileSystem(project=gc_project)
    if cloud_fs.exists(cloud_fp):
        logging.info("Mem-mapped file already exists on GCS, skipping write...")
        return cloud_fp

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
    assert split_name in hfds_splits_set

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
        # inputs = [[bos_id, *e[0:-1]] for e in targets]
        return {"ids": ids}

    ds = ds.map(
        processing_fn,
        batched=True,
        batch_size=hfds_buffer_size * jax.process_count(),
        remove_columns=list(ds.column_names),
    )

    # whatever the official split we're working with happens to be,
    # need to shard by host and drop remainder
    dataset_info = list(hfds.get_dataset_infos(hfds_identifier).values())[0]
    try:
        full_len = dataset_info.splits.get(hfds_split).num_examples
    except AttributeError as e:
        logging.error("You're using a bad dataset, it has no num_examples metadata...")
        raise e
    sharded_full_len = (full_len // pcount) * pcount
    ds = ds.take(sharded_full_len)

    if hfds_splits_set != {"train", "validation", "test"}:
        # if need be, split the training set into train/validation/test
        sharded_val_len = batch_size * 100
        if split_name == "validation":
            sharded_split_len = sharded_val_len
            ds = ds.take(sharded_val_len)
        elif split_name == "test":
            sharded_split_len = sharded_val_len
            ds = ds.skip(sharded_val_len).take(sharded_val_len)
        elif split_name == "train":
            sharded_split_len = sharded_full_len - 2 * sharded_val_len
            ds = ds.skip(2 * sharded_val_len)
        else:
            raise NotImplementedError("Unrecognized split name")
    else:
        sharded_split_len = sharded_full_len

    # we can now guarantee the sharded_split_len is the same on all hosts
    # so make an iterator and write to memmapped file
    n_shard_tokens = sharded_split_len * sequence_len
    n_write_iters = 1024
    logging.debug(f"n_shard_tokens: {n_shard_tokens}")
    logging.debug(f"n_write_iters: {n_write_iters}")

    ds = ds.with_format("numpy")
    ds = ds.iter(batch_size=(n_shard_tokens // n_write_iters), drop_last_batch=True)
    local_fp = posixpath.join("/tmp/", posixpath.split(cloud_fp)[-1])

    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(local_fp, dtype=arr_dtype, mode="w+", shape=(n_shard_tokens,))
    idx = 0
    for _ in tqdm.tqdm(range(n_write_iters), desc=f"Writing {local_fp} with memmap"):
        batch = next(ds)
        logging.debug(f"batch:\n{batch}")
        logging.debug(f"batch.shape: {batch.shape}")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    cloud_fs.upload(local_fp, cloud_fp)
    return cloud_fp


def read_dataset_to_memmmap(
    gc_project: str,
    hfds_identifier: str,
    hftr_tokenizer: hftr.PreTrainedTokenizerFast,
    split_name: str,
    pindex: int,
    workdir: str,
) -> np.ndarray:
    cloud_fp = get_shard_fname(workdir, hfds_identifier, split_name, pindex)
    cloud_fs = gcsfs.GCSFileSystem(project=gc_project)

    local_fp = posixpath.join("/tmp/", posixpath.split(cloud_fp)[-1])
    if not osp.exists(local_fp):
        logging.info(f"Downloading {cloud_fp} to {local_fp}")
        cloud_fs.download(cloud_fp, local_fp)

    logging.info(f"Reading with np.memmap...")
    arr_dtype = get_arr_dtype(hftr_tokenizer.vocab_size)
    arr = np.memmap(local_fp, dtype=arr_dtype, mode="r")
    return arr


def get_dataset(
    gc_project: str,
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
) -> np.ndarray:
    logging.info("Calling write_dataset_to_memmmap...")
    _ = write_dataset_to_memmmap(
        gc_project=gc_project,
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
    logging.info("Calling read_dataset_to_memmmap...")
    arr = read_dataset_to_memmmap(
        gc_project=gc_project,
        hfds_identifier=hfds_identifier,
        hftr_tokenizer=hftr_tokenizer,
        split_name=split_name,
        pindex=pindex,
        workdir=workdir,
    )
    return arr


def get_batch(arr, batch_size, sequence_len, step):  # batch size per host
    # todo: support shuffle indices
    chunk_size = batch_size * sequence_len
    batch = arr[chunk_size * step : chunk_size * (step + 1)]
    batch = np.reshape(batch, [batch_size, sequence_len])
    batch = batch.astype(np.int32)
    return batch


def get_loss_mask(batch, *, pad_token_id, eos_token_id):
    # loss mask that allows training on first occurring eos/pad token as a target,
    # even if eos_token_id == pad_token_id
    loss_mask = np.logical_or(
        np.equal(batch, pad_token_id),
        np.equal(batch, eos_token_id),
    )
    loss_mask = np.logical_not(loss_mask)
    loss_mask = np.pad(loss_mask[:, 0:-1], ((0, 0), (1, 0)), constant_values=True)
    loss_mask = np.cumprod(loss_mask, axis=-1)  # mask everything after the first eos
    return loss_mask
