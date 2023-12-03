import scipy.stats
import os
import librosa
import numpy as np
import pandas as pd


def extract_features(audio_file, target_duration=4, target_sr=44100):
    y, sr = librosa.load(audio_file, sr=target_sr)

    # Ensure audio is target duration
    target_length = int(target_sr * target_duration)
    y = librosa.util.fix_length(y, size=target_length)

    # Normalize amplitude
    y = librosa.util.normalize(y)

    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # spectral feature list
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    melspectogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    crossing_rate = librosa.feature.zero_crossing_rate(y)

    # rhythm feature list
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    fourier_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)

    features = {
        'chroma_stft': np.mean(chroma_stft),
        'chroma_cqt': np.mean(chroma_cqt),
        'chroma_cens': np.mean(chroma_cens),
        'melspectogram': np.mean(melspectogram),
        'rms': np.mean(rms),
        'centroid': np.mean(centroid),
        'bandwidth': np.mean(bandwidth),
        'contrast': np.mean(contrast),
        'flatness': np.mean(flatness),
        'rolloff': np.mean(rolloff),
        'crossing_rate': np.mean(crossing_rate),
        'tempogram': np.mean(tempogram),
        'fourier_tempogram': np.mean(fourier_tempogram)
    }

    for i in range(1, 41):
        features[f'mcffs_{i}'] = np.mean(mfccs[i - 1])
    return features


def process_data(base_dir):
    for folder in os.listdir(base_dir):
        label_list = []
        features_list = []
        fold_dir = os.path.join(base_dir, folder)
        if os.path.isdir(fold_dir):
            for filename in os.listdir(fold_dir):
                file_path = os.path.join(fold_dir, filename)
                if filename.endswith('.wav'):
                    label = filename
                    features = extract_features(file_path)
                    features_list.append(features)
                    label_list.append(label)

        df = pd.DataFrame(features_list)
        df['Label'] = label_list
        df.to_csv('urbansounds_features' + folder + '.csv', index=False)
        # Display the DataFrame
        print(df.head())


def main():
    base_dir = "C:/tmp/sound_datasets/urbansound8k/audio"
    process_data(base_dir)


if __name__ == '__main__':
    main()
