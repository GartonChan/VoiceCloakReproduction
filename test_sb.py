import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs =torchaudio.load('./original_voice.flac')

embeddings = classifier.encode_batch(signal)
print(embeddings.shape)

