# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys
from termcolor import colored

import wandb


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="contrast", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# for wandb log
def get_wandb_name(args):
    wandb_name = "pretrain_"
    if hasattr(args, "n_frames"):
        wandb_name += f"nframe{args.n_frames}_"
    if hasattr(args, "small"):
        if args.small:
            wandb_name += "small_"
    if hasattr(args, "flow_up"):
        if args.flow_up:
            wandb_name += "flow-up_"
    if hasattr(args, "alpha1"):
        wandb_name += f"alpha1-{args.alpha1}_"
    if hasattr(args, "alpha2"):
        wandb_name += f"alpha2-{args.alpha2}_"
    wandb_name += f"crop-{args.crop}_"
    wandb_name += f"aug-{args.aug}_"
    wandb_name += f"{args.dataset}_"
    wandb_name += f"image-size-{args.image_size}_"
    wandb_name += f"l-bn-{args.batch_size}_"
    wandb_name += f"epoch-{args.epochs}_"
    wandb_name = wandb_name.rstrip("_")
    return wandb_name


def init_wandb(args):
    wandb_name = get_wandb_name(args)
    wandb.init(project="PixPro", entity="tomo", name=wandb_name)
    wandb.config.update(args)
    for f in os.listdir(args.output_dir):
        is_file = os.path.isfile(f)
        is_git_file = is_file and "git" in f
        if is_git_file:
            wandb.save(f, base_path=args.output_dir)
