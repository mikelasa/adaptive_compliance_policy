#%%
import os
import json
import torch
import librosa
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
#%%
# force frequency 7k Hz
def read_wrench(episode_path, robot_id):
    force_data = []
    time_data = []
    with open(f'{episode_path}/wrench_data_{robot_id}.json', 'r') as f:
        s = f.read()
        s = s.replace('\t','')
        s = s.replace('\n','')
        s = s.replace(',}','}')
        s = s.replace(',]',']')
        wrench_robot0 = json.loads(s)
        for i in range(len(wrench_robot0)):
            wrench_i = torch.tensor(wrench_robot0[i]['wrench'])
            time_i = wrench_robot0[i]['wrench_time_stamps']
            force_data.append(wrench_i)
            time_data.append(time_i)
    force_data = torch.stack(force_data)
    duration = (time_data[-1] - time_data[0])/1000
    print("time (s):", duration)
    print("frequency (Hz):", len(force_data)/duration)
    return force_data, duration

episode_id, robot_id = '1724793234', 0
episode_path = f'/local/real/liuzeyi/Pyrite/PyriteML/data/2024.08.28.Vase_Wiping_example_episodes/episode_{episode_id}'
wrench_robot, duration = read_wrench(episode_path, robot_id=robot_id)
wrench_robot.shape
#%%
from moviepy.editor import ImageSequenceClip
import torchaudio

def convert_images_to_video(raw_data_path, duration, channel_idx):
    image_folder = f'{raw_data_path}/{channel_idx}'
    images = []
    for current_time in np.arange(0, duration, 0.5):
        images.append(f"{image_folder}/{round(current_time, 1)}.png")

    # Create a video clip from the images
    clip = ImageSequenceClip(images, fps=1/0.5)
    video_filename = f'{raw_data_path}/channel_{channel_idx}.mp4'
    clip.write_videofile(video_filename, codec='libx264')
    
    
def plot_force(save_path, data, sample_rate, current_time, window_size, channel_idx):
    plt.clf()

    plt.figure(figsize=(8, 8))
    start_sample = max(0, int((current_time - window_size / 2) * sample_rate))
    end_sample = min(int((current_time + window_size / 2) * sample_rate), len(data))
    windowed_data = data[start_sample:end_sample]
    
    # run noise reduction
    # TODO: might need to do some processing on the signal, high-pass filter? seems like signals are condensed in low-freq range
    # windowed_audio = nr.reduce_noise(y=windowed_audio, sr=48000, thresh_n_mult_nonstationary=1, stationary=False)
    
    time_axis = np.linspace(
        start_sample / sample_rate, end_sample / sample_rate, len(windowed_data)
    )

    # Plot the waveform in the top subplot
    sr = 7000
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(windowed_data)), windowed_data, color='#ed5c9b')
    # plt.xlabel('')
    plt.xticks([])
    plt.ylabel('Magnitude')
    # plt.ylim(np.min(data), np.max(data))
    # plt.axis('off')

    # audio_transform = torchaudio.transforms.MelSpectrogram(
    #         sample_rate=sr, n_fft=int(sr * 0.025), hop_length=int(sr * 0.01), n_mels=64)
    # mel_spectrogram = audio_transform(windowed_data.float())
    # mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=0.5)
    
    f, t, Sxx = signal.spectrogram(windowed_data, fs=sr, nperseg=512, noverlap=512//4, nfft=1024)

    # Plot the spectrogram in the bottom subplot
    plt.subplot(2, 1, 2)
    # librosa.display.specshow(mel_spectrogram_db, x_axis='time', y_axis='mel', sr=sr, hop_length=int(sr * 0.01), cmap='magma')
    # plt.clim(-20, 40)
    # plt.colorbar(format='%+2.0f dB')
    fmin = 0 # Hz
    fmax = 200 # Hz
    freq_slice = np.where((f >= fmin) & (f <= fmax))
    # keep only frequencies of interest
    f   = f[freq_slice]
    Sxx = Sxx[freq_slice,:][0]
    plt.pcolormesh(t, f, Sxx, shading='gouraud')

    # Adjust spacing between subplots
    plt.tight_layout()
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    # plt.axis('off')
    os.system(f'mkdir -p {save_path}/{channel_idx}')
    plt.savefig(f"{save_path}/{channel_idx}/{current_time}.png")
    return Sxx.min(), Sxx.max()
#%%
sr = 7000
window_size = 1
global_min, global_max = np.inf, -np.inf
save_path = f'/local/real/liuzeyi/Pyrite/PyriteML/local/episode_{episode_id}/robot_{robot_id}'
for current_time in np.arange(0, duration, 0.5):
    for channel_idx in range(wrench_robot.shape[1]):
        Sxx_min, Sxx_max = plot_force(save_path, wrench_robot[:, channel_idx], sr, round(current_time, 1), window_size=window_size, channel_idx=channel_idx)
        if Sxx_min < global_min:
            global_min = Sxx_min
        if Sxx_max > global_max:
            global_max = Sxx_max
print(global_min, global_max)
#%%
save_path = f'/local/real/liuzeyi/Pyrite/PyriteML/local/episode_{episode_id}/robot_{robot_id}'
for channel_idx in range(wrench_robot.shape[1]):
    convert_images_to_video(save_path, duration=duration, channel_idx=channel_idx)
    
#%% OLD CODE
# # 100 Hz
# force = torch.randn(16, 300, 6)
# x = force[:, :, 0]
# # transform = torchaudio.transforms.Spectrogram(n_fft=34)
# # spec = transform(x)[0]
# spec = torch.tensor(signal.spectrogram(x, fs=300, nperseg=24, noverlap=3, nfft=24)[2])[0]
# print(spec.shape)
# padded_spec = torch.zeros((32, 32))
# x_margin = (padded_spec.shape[0] - spec.shape[0])//2
# y_margin = (padded_spec.shape[1] - spec.shape[1])//2
# padded_spec[1+x_margin:padded_spec.shape[0]-x_margin, y_margin:padded_spec.shape[1]-y_margin] = spec
# padded_spec.shape
# #%%
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.imshow(padded_spec, cmap='inferno')
# plt.title('Spectrogram Vis')
# plt.ylabel('Frequency bins')
# plt.xlabel('Time frames')
# plt.show()
# # %%
# import timm
# model = timm.create_model('resnet18.a1_in1k', pretrained=False, global_pool='', num_classes=0)
# model
# %%
