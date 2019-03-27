import wave
import os
from pydub import AudioSegment

def ShortenAudio(filename):
    indicators = filename.split("-")
    if indicators[1] == "01":
        return shorten_speech(filename)
    else:
        return shorten_song(filename)

def shorten_song(filename):
    song = AudioSegment.from_wav("TrainingSet\\" + filename)
    start_sec = int(1 * 1000)
    end_sec = int(4 * 1000)
    cut_audio = song[start_sec:end_sec]

    return cut_audio

def shorten_speech(filename):
    song = AudioSegment.from_wav("TrainingSet\\" + filename)
    start_sec = int(0.25 * 1000)
    end_sec = int(3.25 * 1000)
    cut_audio = song[start_sec:end_sec]

    return cut_audio

if __name__ == "__main__":
    if not os.path.exists("cut-audio"):
        os.mkdir("cut_audio")
        for filename in os.listdir("TrainingSet"):
            return_file = ShortenAudio(filename)
            return_file.export("cut_audio\\cut-" + filename, "wav")
