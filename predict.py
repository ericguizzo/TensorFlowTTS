import os
import cog
import json
import tempfile
import zipfile
from pathlib import Path
import tensorflow as tf
import soundfile as sf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import AutoProcessor


class TTS(cog.Predictor):
    def setup(self):
        """Load the fastspeech and melgan models """
        # initialize fastspeech2 model.
        self.fastspeech2 = TFAutoModel.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")
        # initialize mb_melgan model
        self.mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")
        # inference
        self.processor = AutoProcessor.from_pretrained("tensorspeech/tts-fastspeech2-ljspeech-en")

    @cog.input("input", type=str, help="String to be converted to speech audio")
    @cog.input("speaker_ids", type=int, default=0)
    @cog.input("speed_ratios", type=float, default=1.0)
    @cog.input("f0_ratios", type=float, default=1.0)
    @cog.input("energy_ratios", type=float, default=1.0)

    def predict(self, input, speaker_ids, speed_ratios, f0_ratios, energy_ratios):
        """Compute TTS on input string"""
        # inference
        input_ids = self.processor.text_to_sequence(input)

        # fastspeech inference
        mel_before, mel_after, duration_outputs, _, _ = self.fastspeech2.inference(
            input_ids=tf.expand_dims(tf.convert_to_tensor([input_id], dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([speaker_id], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([speed_ratio], dtype=tf.float32),
            f0_ratios =tf.convert_to_tensor([f0_ratio], dtype=tf.float32),
            energy_ratios =tf.convert_to_tensor([energy_ratio], dtype=tf.float32),
        )

        # melgan inference
        audio_before = self.mb_melgan.inference(mel_before)[0, :, 0]
        audio_after =self. mb_melgan.inference(mel_after)[0, :, 0]

        output_path = Path('output').mkdir(parents=True, exist_ok=True)

        out_path = Path(tempfile.mkdtemp())
        zip_path = Path(tempfile.mkdtemp()) / "output.zip"

        out_path_before = out_path / 'audio_before.wav'
        out_path_after = out_path / 'audio_after.wav'

        # save to file
        sf.write(str(out_path_before), audio_before, 22050, "PCM_16")
        sf.write(str(out_path_after), audio_after, 22050, "PCM_16")

        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.write(str(out_path_before))
            zf.write(str(out_path_after))

        return zip_path
