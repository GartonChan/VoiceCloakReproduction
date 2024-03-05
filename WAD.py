import os
from speechbrain.pretrained import EncoderDecoderASR
from jiwer import wer

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", 
                                           savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                           run_opts={"device":"cuda"})

def transcribe_audio(audio_path):
    words = asr_model.transcribe_file(audio_path)
    return words

original_audio_folder = os.getcwd() + "/test-clean/"
processed_audio_folder = os.getcwd() + "/processed-test-clean/"

for root, dirs, files in os.walk(original_audio_folder):
    if any(file.endswith(".flac") for file in files):
        parts = root.split(os.sep)
        if len(parts) < 3:
            continue
        speaker_id, recording_id = parts[-2], parts[-1]
        txt_filename = f"{speaker_id}-{recording_id}.trans.txt"
        txt_path = os.path.join(root, txt_filename)

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.read().strip().split('\n')
                text_dict = {line.split(' ')[0]: ' '.join(line.split(' ')[1:]) for line in lines}
        else:
            print(f"Text file not found for audio in {root}")
            continue

        for file in files:
            if file.endswith('.flac'):
                utterance_id = file.split('.')[0]
                if utterance_id in text_dict:
                    original_text = text_dict[utterance_id]
                else:
                    print(f"Transcription not found for {file}")
                    continue

                original_audio_path = os.path.join(root, file)
                processed_audio_filename = 'processed_' + file
                processed_audio_path = os.path.join(processed_audio_folder, processed_audio_filename)

                original_transcribed_text = transcribe_audio(original_audio_path)
                processed_transcribed_text = transcribe_audio(processed_audio_path)
                
                original_wer = wer(original_text, original_transcribed_text)
                processed_wer = wer(original_text, processed_transcribed_text)
                
                WAD = processed_wer - original_wer
                
                with open('WAD_results.txt', 'a') as f:
                    print(f"Original audio: {file}", file=f)
                    print(f"Original word error rate: {original_wer}", file=f)
                    print(f"Processed word error rate: {processed_wer}", file=f)
                    print("WAD: ", WAD, file=f)
                    print("", file=f)
