from pandas.core.frame import DataFrame
from matplotlib.pyplot import MultipleLocator
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wave
import pyaudio
import librosa
import librosa.display
import pandas as pd
import numpy as np
import copy

XTICKS_SPACE=30

class Singya(object):
    def __init__(self, song_wav="./song.wav", channels=1, format=pyaudio.paInt16, rate=22050, record_time=5) -> None:
        super().__init__()
        self.song_wav = song_wav
        self.channels = channels
        self.format = format
        self.rate = rate
        self.chunk = int(self.rate/20) # RATE/number of updates per second
        self.note_sample_step = int(self.rate/3000)
        self.record_time = record_time
        self.data = None
        self.chromagram = None
        self.note_values = None
        self.note_values_with_octave = None
        self.pitches = None
        self.magnitudes = None


    def record_song(self):
        p = pyaudio.PyAudio()
        stream = p.open(
            format = self.format,
            channels = self.channels,
            rate = self.rate,
            input = True,
            frames_per_buffer = self.chunk)

        print("***Start recording***")
        frames = []
        for _ in range(int(self.record_time*self.rate/self.chunk)):
            data = stream.read(self.chunk)
            frames.append(data)
        print("***End recording***")

        stream.stop_stream()
        stream.close()
        p.terminate()

        print("***Start writing output***")
        wf = wave.open(self.song_wav, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("***End writing output***")


    def load_song(self, file_path="", sr=0):
        if not file_path or not sr:
            file_path = self.song_wav
            sr = self.rate
        
        data, sr = librosa.load(file_path, self.rate)
        
        data_harm = librosa.effects.harmonic(data, margin=8)
        chromagram = librosa.feature.chroma_cqt(data_harm, sr=sr)
        chroma_df = pd.DataFrame(chromagram)
        chroma_detected = (chroma_df==1).astype(int)
        labels=np.array(range(1,13))

        # save variables to self
        self.data = data
        self.chromagram = chromagram
        self.note_values = labels.dot(chroma_detected)
        self.pitches, self.magnitudes = librosa.piptrack(y=data, sr=sr)

        # calculate pitch based on self.note_values and detect_pitch
        self._set_note_octave(n_smooth=3)

    
    def _gen_y_labels(self):
        n_octave = 7 # chroma_cqt supports 7 octaves by default
        notes = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        octave = range(1, n_octave+1)
        y_labels = []
        for oct in octave:
            for note in notes:
                y_labels.append(note+str(oct))
        return y_labels


    def _detect_pitch(self, pitches, magnitudes, t):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        return pitch


    def _pitch_hz_to_note(self):
        notes = []
        check_idx = 0
        while check_idx < self.pitches.shape[1]:
            notes.append(librosa.hz_to_note(self._detect_pitch(self.pitches, self.magnitudes, check_idx)))
            check_idx += 1
        return notes

    
    # set self.note_value_with_octave based on majority
    def _set_note_octave(self, n_smooth):
        notes = self._pitch_hz_to_note()
        octaves = [int(note[-1]) for note in notes]
        # print("octaves: ", len(octaves), octaves)
        # print("note_values: ", len(self.note_values), self.note_values)
        self.note_values_with_octave = copy.deepcopy(self.note_values)

        left, right = 0, 1
        curr_note = self.note_values[0]
        flag = 0

        while right < len(self.note_values):
            if self.note_values[right] == curr_note:
                right += 1
                flag = 0
            else:
                # smooth the note_values_with_octave, omit the notes chunck with length<=n
                if right-left <= n_smooth:
                    left_value = -1
                    if left > 0:
                        left_value = self.note_values_with_octave[left-1]

                    if left_value != -1:
                        self.note_values_with_octave[left:right] = [left_value]*(right-left)
                        # print("smooth: ", left, right, self.note_values_with_octave[left:right], self.note_values_with_octave[left-1])
                        curr_note = self.note_values[right]
                        left = right
                        right = left + 1
                        flag = 1
                        continue

                maj_octave = stats.mode(octaves[left:right])[0][0]
                self._calculate_note_octave(left, right, maj_octave)

                curr_note = self.note_values[right]
                left = right
                right = left + 1
                flag = 1
        
        if not flag:
            if right-left <= n_smooth:
                left_value = self.note_values_with_octave[left-1]
                self.note_values_with_octave[left:right] = [left_value]*(right-left)
            maj_octave = stats.mode(octaves[left:right])[0][0]
            self._calculate_note_octave(left, right, maj_octave)

        # print("note_values_with_octave: ", len(self.note_values_with_octave), self.note_values_with_octave)
    

    def _calculate_note_octave(self, left, right, octave):
        # print("ori: ", self.note_values_with_octave[left:right])
        # print("to set: ", [self.note_values[left] + 12 * octave] * (right-left))
        self.note_values_with_octave[left:right] = \
            [self.note_values[left] + 12 * octave] * (right-left)
    

    def _smooth_notes(self, n_smooth=0):

        return 


    def draw_chromagram(self):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        librosa.display.specshow(self.chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
        ax.yaxis.set_major_formatter(librosa.display.NoteFormatter())
        plt.show()


    # Recommended
    def draw_wave(self):
        plt.figure(figsize=(15, 5))
        librosa.display.waveplot(self.data, sr=self.rate, color="#ff9999")
        plt.show()
    

    def draw_scatter(self):
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(111)
        ax.scatter(range(len(self.note_values)), self.note_values, marker="s", s=1, color="red")

        plt.grid(linewidth=0.5)
        plt.xlim(0, self.pitches.shape[1])
        plt.yticks(range(1,13), ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])
        x_coords = librosa.core.frames_to_time(np.arange(self.pitches.shape[1] + 1), sr=self.rate)
        x_coords = [format(x, ".1f") for x in x_coords]
        plt.xticks(np.arange(self.pitches.shape[1] + 1)[::XTICKS_SPACE], x_coords[::XTICKS_SPACE])


    # Recommended
    def draw_hlines(self):
        y_min, y_max = min(self.note_values_with_octave)-1, max(self.note_values_with_octave)+1
        width, height= self.record_time*3, (y_max-y_min)/2
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(111)

        ax.plot(self.note_values_with_octave, color="#008080", linewidth=0.8)
        ax.hlines(self.note_values_with_octave, range(len(self.note_values_with_octave)), np.array(range(len(self.note_values_with_octave)))+1, color="red", linewidth=10)

        plt.grid(linewidth=0.5)
        plt.xlim(0, self.pitches.shape[1])
        y_labels = self._gen_y_labels()
        plt.yticks(range(1, len(y_labels)+1), y_labels)
        
        # ax.yaxis.set_major_locator(MultipleLocator(10))
        plt.ylim(y_min, y_max)

        x_coords = librosa.core.frames_to_time(np.arange(self.pitches.shape[1] + 1), sr=self.rate)
        x_coords = [format(x, ".1f") for x in x_coords]
        plt.xticks(np.arange(self.pitches.shape[1] + 1)[::XTICKS_SPACE], x_coords[::XTICKS_SPACE])


