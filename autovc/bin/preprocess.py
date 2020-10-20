#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
# Copyright 2020 Jaegeon Jo
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os
import re

import librosa
import numpy as np
import soundfile as sf
import yaml

from tqdm import tqdm

from parallel_wavegan.datasets import AudioDataset
from parallel_wavegan.datasets import AudioSCPDataset
from parallel_wavegan.utils import write_hdf5
from parallel_wavegan.bin.preprocess import logmelfilterbank

def speaker_embedding(spk,
                      spkList,
                      spk_emb_dim=256,
                      spk_emb_type="onehot"
                      ):
    """Embed speaker information

    Args:
        spk (str): name of speaker.
        spkList (List[str]): List of speakers.
        spk_emb_dim (int): dimension of spekeaker embedding array
        spk_emb_type (str): Type of embedding 
    
    Returns:
        ndarray: Speaker embedding (spk_emb_dim)
    
    """

    assert spk in spkList, f"{spk} is not in speaker list."

    if "onehot" in spk_emb_type:
        spk_emb = np.zeros(spk_emb_dim)
        spk_emb[spkList.index(spk)] = 1.0
        return spk_emb
    else:
        raise ValueError(f"{spk_emb_type} is not supported for speaker embedding.")
        


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py).")
    parser.add_argument("--wav-scp", "--scp", default=None, type=str,
                        help="kaldi-style wav.scp file. you need to specify either scp or rootdir.")
    parser.add_argument("--segments", default=None, type=str,
                        help="kaldi-style segments file. if use, you must to specify both scp and segments.")
    parser.add_argument("--rootdir", default=None, type=str,
                        help="directory including wav files. you need to specify either scp or rootdir.")
    parser.add_argument("--dumpdir", type=str, required=True,
                        help="directory to dump feature files.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--spkList", "--spks", type=str, required=True,
                        help="list of speakers. default is all speakers in dumpdir")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check arguments
    if (args.wav_scp is not None and args.rootdir is not None) or \
            (args.wav_scp is None and args.rootdir is None):
        raise ValueError("Please specify either --rootdir or --wav-scp.")

    # get dataset
    if args.rootdir is not None:
        dataset = AudioDataset(
            args.rootdir, "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )
    else:
        dataset = AudioSCPDataset(
            args.wav_scp,
            segments=args.segments,
            return_utt_id=True,
            return_sampling_rate=True,
        )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    # process each data
    spkList = re.split(" |\n", args.spkList)

    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        if utt_id.split("_")[0] not in spkList:
            continue
        assert len(audio.shape) == 1, \
            f"{utt_id} seems to be multi-channel signal."
        assert np.abs(audio).max() <= 1.0, \
            f"{utt_id} seems to be different from 16 bit PCM."
        assert fs == config["sampling_rate"], \
            f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(audio,
                                            top_db=config["trim_threshold_in_db"],
                                            frame_length=config["trim_frame_size"],
                                            hop_length=config["trim_hop_size"])

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]
        else:
            # NOTE(kan-bayashi): this procedure enables to train the model with different
            #   sampling rate for feature and audio, e.g., training with mel extracted
            #   using 16 kHz audio and 24 kHz audio as a target waveform
            x = librosa.resample(audio, fs, config["sampling_rate_for_feats"])
            sampling_rate = config["sampling_rate_for_feats"]
            assert config["hop_size"] * config["sampling_rate_for_feats"] % fs == 0, \
                "hop_size must be int value. please check sampling_rate_for_feats is correct."
            hop_size = config["hop_size"] * config["sampling_rate_for_feats"] // fs

        # extract feature
        mel = logmelfilterbank(x,
                               sampling_rate=sampling_rate,
                               hop_size=hop_size,
                               fft_size=config["fft_size"],
                               win_length=config["win_length"],
                               window=config["window"],
                               num_mels=config["num_mels"],
                               fmin=config["fmin"],
                               fmax=config["fmax"])

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")
        audio = audio[:len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(f"{utt_id} causes clipping. "
                         f"it is better to re-consider global gain scale.")
            continue

        # compute speaker embedding
        spk = utt_id.split("_")[0]
        spk_emb = speaker_embedding(spk, 
                                    spkList,
                                    spk_emb_dim = config["spk_emb_dim"],
                                    spk_emb_type = config["spk_emb_type"])

        # save
        if config["format"] == "hdf5":
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "wave", audio.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "feats", mel.astype(np.float32))
            write_hdf5(os.path.join(args.dumpdir, f"{utt_id}.h5"), "spkemb", spk_emb.astype(np.float32))
        elif config["format"] == "npy":
            np.save(os.path.join(args.dumpdir, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{utt_id}-feats.npy"),
                    mel.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(args.dumpdir, f"{utt_id}-spkemb.npy"),
                    spk_emb.astype(np.float32), allow_pickle=False)
        else:
            raise ValueError("support only hdf5 or npy format.")
    logging.warning(f"Embedded speakers = {spkList}")


if __name__ == "__main__":
    main()
