import numpy as np
import torch
import torchaudio
import os
import json
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


def generate_one_hot_label(idx, size):
    oneHotLabel = [0 for i in range(size)]
    oneHotLabel[idx] = 1
    return np.array(oneHotLabel)

# # Get the embedding
# model = load_model('/home/gartonchan/src/voiceCloak/ResCNN_triplet_training_checkpoint_265.h5')
# embedding = predict_embedding(model, "./samples/PhilippeRemy/PhilippeRemy_001.wav")
# print(embedding)
# print(np.shape(embedding))
# print(type(embedding))

# # Generate label
# oneHotLabel = generate_one_hot_label(2, 512)
# # Append label
# print(np.reshape(np.append(embedding, oneHotLabel), (2, 512)))

# Extract embedding and concatenate with label
identity_set = set()
for root, dirs, files in os.walk("/home/gartonchan/tensorflow_datasets/downloads/LibriSpeech/train-clean-100/"):
    for file in files:
        print(file)
        strs = file.split('-')
        identity_set.add(strs[0])
print(identity_set)

for root, dirs, files in os.walk("/home/gartonchan/tensorflow_datasets/downloads/extracted/test-clean/"):
    for file in files:
        print(file)
        strs = file.split('-')
        if strs[0] in identity_set:
            continue
        identity_set.add(strs[0])
print(identity_set)

identity_list = list(identity_set)
identity_list.sort()

print(identity_list)
print(len(identity_list))
print(identity_list.index('6818'))

train_set = []
for root, dirs, files in os.walk("/home/gartonchan/tensorflow_datasets/downloads/LibriSpeech/train-clean-100/"):
    for file in files:
        if file.split('.')[-1] != 'flac':
            continue
        strs = file.split('-')
        # if strs[0] != '19':
        #     continue
        identity_idx = identity_list.index(strs[0])
        file_path = os.path.join(root, file)
        # get embedding
        signal, fs =torchaudio.load(file_path)
        embedding = classifier.encode_batch(signal)
        # generate one-hot with idx
        label = generate_one_hot_label(identity_idx%192, 192)
        # concatenate with label
        e_y = np.reshape(np.append(embedding, label), (2, 192))
        print(e_y)
        e_y_list_fmt = e_y.tolist()
        # print(e_y[1][identity_idx])
        train_set.append(e_y_list_fmt)
        print("data_size = ", len(train_set))
        
json_str = json.dumps(train_set)
with open("train_set.json", "w") as f:
    # json.dump(data_set, f)
    f.write(json_str)
    
test_set = []
for root, dirs, files in os.walk("/home/gartonchan/tensorflow_datasets/downloads/extracted/test-clean/LibriSpeech/test-clean/"):
    for file in files:
        if file.split('.')[-1] != 'flac':
            continue
        strs = file.split('-')
        identity_idx = identity_list.index(strs[0])
        file_path = os.path.join(root, file)
        # get embedding
        signal, fs =torchaudio.load(file_path)
        embedding = classifier.encode_batch(signal)
        # generate one-hot with idx
        label = generate_one_hot_label(identity_idx%192, 192)
        # concatenate with label
        e_y = np.reshape(np.append(embedding, label), (2, 192))
        # print(e_y[1][identity_idx])
        e_y_list_fmt = e_y.tolist()
        test_set.append(e_y_list_fmt)
        print("data_size = ", len(test_set))
print(test_set)
json_data = json.dumps(test_set)
with open("test_set.json", "w") as f:
    # json.dump(data_set, f)
    f.write(json_data)
    
with open("log.txt", "w") as f:
    f.write("train_set_size = {}\n".format(len(train_set)))
    f.write("test_set_size = {}\n".format(len(test_set)))