import json
import os
import tempfile
import zipfile
from pathlib import Path

import cog
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
import yaml
from tensorflow_tts.inference import AutoConfig, AutoProcessor, TFAutoModel


class TTS(cog.Predictor):
    def setup(self):
        """Load the fastspeech and melgan models"""
        # initialize fastspeech2 model.
        self.fastspeech2 = TFAutoModel.from_pretrained(
            "tensorspeech/tts-fastspeech2-ljspeech-en"
        )
        # initialize mb_melgan model
        self.mb_melgan = TFAutoModel.from_pretrained(
            "tensorspeech/tts-mb_melgan-ljspeech-en"
        )
        # inference
        self.processor = AutoProcessor.from_pretrained(
            "tensorspeech/tts-fastspeech2-ljspeech-en"
        )

    @cog.input("input", type=str, help="String to be converted to speech audio")
    @cog.input("speaker_id", type=int, default=0)
    @cog.input("speed_ratio", type=float, default=1.0)
    @cog.input("f0_ratio", type=float, default=1.0)
    @cog.input("energy_ratio", type=float, default=1.0)
    def predict(self, input, speaker_id, speed_ratio, f0_ratio, energy_ratio):
        """Compute TTS on input string"""
        # inference
        input_id = self.processor.text_to_sequence(input)

        # fastspeech inference
        mel_before, mel_after, duration_outputs, _, _ = self.fastspeech2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor(input_id, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([speaker_id], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([speed_ratio], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([f0_ratio], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([energy_ratio], dtype=tf.float32),
        )

        # melgan inference
        audio_after = self.mb_melgan.inference(mel_after)[0, :, 0]

        # save to file
        output_path = Path("output").mkdir(parents=True, exist_ok=True)
        out_path = Path(tempfile.mkdtemp()) / "output.wav"
        sf.write(str(out_path), audio_after, 22050, "PCM_16")

        return out_path
