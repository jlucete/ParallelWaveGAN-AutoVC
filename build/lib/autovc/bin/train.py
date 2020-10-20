#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Train AutoVC."""

import argparse
import logging
import os
import sys

from collections import defaultdict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import yaml

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import autovc

from autovc.datasets import AudioMelDataset
from autovc.datasets import AudioMelSCPDataset
from parallel_wavegan.utils import read_hdf5

import autovc.models
import autovc.optimizers

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for AutoVC training."""

    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=torch.device("cpu"),
                 ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers. It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        x_real = x[-2]
        emb_org = x[-1]

        #######################
        #      Generator      #
        #######################
        x_real = x_real.transpose(1,2)
        # Identity mapping loss
        x_identic, x_identic_psnt, code_real = self.model["generator"](x_real, emb_org, emb_org)
        g_loss_id = self.criterion["mse"](x_real, x_identic)
        g_loss_id_psnt = self.criterion["mse"](x_real, x_identic_psnt)
        self.total_train_loss["train/identity_mapping_loss"] += g_loss_id.item()
        self.total_train_loss["train/identity_mapping_loss_psnt"] += g_loss_id_psnt.item()

        # Code Semantic loss
        code_reconst = self.model["generator"](x_identic_psnt, emb_org, None)
        g_loss_cd = self.criterion["l1"](code_real, code_reconst)
        self.total_train_loss["train/code_semantic_loss"] += g_loss_cd.item()
        
        # Total generator loss
        g_loss = g_loss_id + g_loss_id_psnt + self.config["lambda_cd"] * g_loss_cd
        self.total_train_loss["train/generator_loss"] += g_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        g_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)

        x_real = x[-2]
        emb_org = x[-1]

        #######################
        #      Generator      #
        #######################
        x_real = x_real.transpose(1,2)
        x_identic, x_identic_psnt, code_real = self.model["generator"](x_real, emb_org, emb_org)

        # Identity mapping loss
        g_loss_id = self.criterion["mse"](x_real, x_identic)
        g_loss_id_psnt = self.criterion["mse"](x_real, x_identic_psnt)
        self.total_eval_loss["eval/identity_mapping_loss"] += g_loss_id.item()
        self.total_eval_loss["eval/identity_mapping_loss_psnt"] += g_loss_id_psnt.item()

        # Code Semantic loss
        code_reconst = self.model["generator"](x_identic_psnt, emb_org, None)
        g_loss_cd = self.criterion["l1"](code_real, code_reconst)
        self.total_eval_loss["eval/code_semantic_loss"] += g_loss_cd.item()
        
        # Total generator loss
        g_loss = g_loss_id + g_loss_id_psnt + self.config["lambda_cd"] * g_loss_cd
        self.total_eval_loss["eval/generator_loss"] += g_loss.item()

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # FIXME(Jaegeon): implement ParallelWaveGAN-AutoVC end-to-end intermediate result.
        '''
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch)

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), y,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), y_,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break
        '''

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(self,
                 batch_max_steps=20480,
                 hop_size=256,
                 aux_context_window=2,
                 use_noise_input=False,
                 len_crop=128,
                 ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of auxiliary features.
            aux_context_window (int): Context window size for auxiliary feature conv.
            use_noise_input (bool): Whether to use noise input.
            len_crop (int): Length of single crop.

        """
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size
        self.aux_context_window = aux_context_window
        self.use_noise_input = use_noise_input
        self.len_crop = len_crop

        # set useful values in random cutting
        self.start_offset = aux_context_window
        self.end_offset = -(self.batch_max_frames + aux_context_window)
        self.mel_threshold = self.batch_max_frames + 2 * aux_context_window

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio, features and speaker embedding.

        Returns:
            Tensor: Gaussian noise batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T'), where
                T = (T' - 2 * aux_context_window) * hop_size.
            Tensor: Speaker Embedding (B, spk_emb_dim, 1)
            Tensor: Target signal batch (B, 1, T).

        """
        # check length
        batch = [self._adjust_length(*b) for b in batch if len(b[1]) > self.mel_threshold]
        xs, cs, es = [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        y_batch = [x[start: end] for x, start, end in zip(xs, x_starts, x_ends)]
        c_batch = []
        for tmp in cs:
            tmp = np.array(tmp)
            if tmp.shape[0] < self.len_crop:
                len_pad = self.len_crop - tmp.shape[0]
                c_batch.append(np.pad(tmp, ((0,len_pad),(0,0)), 'constant'))
            elif tmp.shape[0] > self.len_crop:
                left = np.random.randint(tmp.shape[0]-self.len_crop)
                c_batch.append(tmp[left:left+self.len_crop, :])
            else:
                c_batch.append(tmp)

        e_batch = es

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = torch.tensor(y_batch, dtype=torch.float).unsqueeze(1)  # (B, 1, T)
        c_batch = torch.tensor(c_batch, dtype=torch.float).transpose(2, 1)  # (B, C, T')
        e_batch = torch.tensor(e_batch, dtype=torch.float) # (B, E, 1)

        # make input noise signal batch tensor
        if self.use_noise_input:
            z_batch = torch.randn(y_batch.size())  # (B, 1, T)
            return (z_batch, c_batch, e_batch), y_batch
        else:
            return (c_batch, e_batch,), y_batch

    def _adjust_length(self, x, c, e):
        """Adjust the audio and feature lengths.

        Note:
            Basically we assume that the length of x and c are adjusted
            through preprocessing stage, but if we use other library processed
            features, this process will be needed.

        """
        if len(x) < len(c) * self.hop_size:
            x = np.pad(x, (0, len(c) * self.hop_size - len(x)), mode="edge")

        # check the legnth is valid
        assert len(x) == len(c) * self.hop_size

        return x, c, e


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel WaveGAN (See detail in parallel_wavegan/bin/train.py).")
    parser.add_argument("--train-wav-scp", default=None, type=str,
                        help="kaldi-style wav.scp file for training. "
                             "you need to specify either train-*-scp or train-dumpdir.")
    parser.add_argument("--train-feats-scp", default=None, type=str,
                        help="kaldi-style feats.scp file for training. "
                             "you need to specify either train-*-scp or train-dumpdir.")
    parser.add_argument("--train-spkemb-scp", default=None, type=str,
                        help="kaldi-style spkemb.scp file for training. "
                             "you need to specify either train-*-scp or train-dumpdir.")
    parser.add_argument("--train-segments", default=None, type=str,
                        help="kaldi-style segments file for training.")
    parser.add_argument("--train-dumpdir", default=None, type=str,
                        help="directory including training data. "
                             "you need to specify either train-*-scp or train-dumpdir.")
    parser.add_argument("--dev-wav-scp", default=None, type=str,
                        help="kaldi-style wav.scp file for validation. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--dev-feats-scp", default=None, type=str,
                        help="kaldi-style feats.scp file for vaidation. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--dev-spkemb-scp", default=None, type=str,
                        help="kaldi-style spkemb.scp file for vaidation. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--dev-segments", default=None, type=str,
                        help="kaldi-style segments file for validation.")
    parser.add_argument("--dev-dumpdir", default=None, type=str,
                        help="directory including development data. "
                             "you need to specify either dev-*-scp or dev-dumpdir.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if (args.train_feats_scp is not None and args.train_dumpdir is not None) or \
            (args.train_feats_scp is None and args.train_dumpdir is None):
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.")
    if (args.dev_feats_scp is not None and args.dev_dumpdir is not None) or \
            (args.dev_feats_scp is None and args.dev_dumpdir is None):
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.")
    if (args.train_spkemb_scp is not None and args.train_dumpdir is not None) or \
            (args.train_spkemb_scp is None and args.train_dumpdir is None):
        raise ValueError("Please specify either --train-dumpdir or --train-*-scp.")
    if (args.dev_spkemb_scp is not None and args.dev_dumpdir is not None) or \
            (args.dev_spkemb_scp is None and args.dev_dumpdir is None):
        raise ValueError("Please specify either --dev-dumpdir or --dev-*-scp.")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = autovc.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    if config["remove_short_samples"]:
        #FIXME(Jaegeon Jo): This is not for AutoVC
        mel_length_threshold = config["batch_max_steps"] // config["hop_size"] + \
            2 * config["autovc_generator_params"].get("aux_context_window", 0)
    else:
        mel_length_threshold = None
    if args.train_wav_scp is None or args.dev_wav_scp is None:
        if config["format"] == "hdf5":
            audio_query, mel_query, spkemb_query = "*.h5", "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
            spkemb_load_fn = lambda x: read_hdf5(x, "spkemb")
        elif config["format"] == "npy":
            audio_query, mel_query, spkemb_query = "*-wave.npy", "*-feats.npy", "*-spkemb.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load
            spkemb_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")
    if args.train_dumpdir is not None:
        train_dataset = AudioMelDataset(
            root_dir=args.train_dumpdir,
            audio_query=audio_query,
            mel_query=mel_query,
            spkemb_query=spkemb_query,
            audio_load_fn=audio_load_fn,
            mel_load_fn=mel_load_fn,
            spkemb_load_fn=spkemb_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    else:
        train_dataset = AudioMelSCPDataset(
            wav_scp=args.train_wav_scp,
            feats_scp=args.train_feats_scp,
            spkemb_scp=args.train_spkemb_scp,
            segments=args.train_segments,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    if args.dev_dumpdir is not None:
        dev_dataset = AudioMelDataset(
            root_dir=args.dev_dumpdir,
            audio_query=audio_query,
            mel_query=mel_query,
            spkemb_query=spkemb_query,
            audio_load_fn=audio_load_fn,
            mel_load_fn=mel_load_fn,
            spkemb_load_fn=spkemb_load_fn,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    else:
        dev_dataset = AudioMelSCPDataset(
            wav_scp=args.dev_wav_scp,
            feats_scp=args.dev_feats_scp,
            spkemb_scp=args.dev_spkemb_scp,
            segments=args.dev_segments,
            mel_length_threshold=mel_length_threshold,
            allow_cache=config.get("allow_cache", False),  # keep compatibility
        )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
        # keep compatibility
        aux_context_window=config["autovc_generator_params"].get("aux_context_window", 0),
        # keep compatibility
        use_noise_input=config.get(
            "generator_type", "ParallelWaveGANGenerator") != "MelGANGenerator",
        len_crop=config["len_crop"],
    )
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models and optimizers
    generator_class = getattr(
        autovc.models,
        # keep compatibility
        config.get("generator_type", "AutoVCGenerator"),
    )
    setattr(generator_class, 'num_mels', config["num_mels"])
    model = {
        "generator": generator_class(
            **config["autovc_generator_params"]).to(device),
    }
    criterion = {
        "l1": torch.nn.L1Loss().to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    generator_optimizer_class = getattr(
        autovc.optimizers,
        # keep compatibility
        config.get("generator_optimizer_type", "RAdam"),
    )
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("generator_scheduler_type", "StepLR"),
    )
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
    }
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
    logging.info(model["generator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
