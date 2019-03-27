import os
import shutil
import math
import random
from pydub import AudioSegment

def get_data_samples(path_dataset):
    count = 0
    num_actors = 24

    for actor in range(1,num_actors+1):
        wav_files_song = []
        wav_files_speech = []

        # numbers below 10 start with "0"
        if actor < 10:
            actor = "0" + str(actor)
        actor = str(actor)

        #Actor 18 has no song files
        if not actor == "18":
            #files in the actor's directory for songs directory
            files_song = os.listdir("./Audio_Song_Actors_01-24/Actor_{}".format(actor))
            files_song_id = [(f,f.rstrip(".wav").split("-")) for f in files_song]

            for file,id in files_song_id:
                #03 = happy, 04 = sad, 05 = angry
                #id[2] is the number that determines the emotion the actor was speaking with in this video
                if int(id[2]) in (3,4,5):
                    wav_files_song.append(file)
                    count += 1

            # move files to Dataset directory
            for file in wav_files_song:
                shutil.move("./Audio_Song_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)


        # files in the actor's directory for speech directory
        files_speech = os.listdir("./Audio_Speech_Actors_01-24/Actor_{}".format(actor))
        files_speech_id = [(f,f.rstrip(".wav").split("-")) for f in files_speech]

        for file,id in files_speech_id:
            #03 = happy, 04 = sad, 05 = angry
            if int(id[2]) in (3,4,5):
                wav_files_speech.append(file)
                count += 1

        # move files to Dataset directory
        for file in wav_files_speech:
            shutil.move("./Audio_Speech_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)

def shorten_audio(filename, path_dataset):
    indicators = filename.split("-")
    if indicators[1] == "01":
        return shorten_speech(filename, path_dataset)
    else:
        return shorten_song(filename, path_dataset)

def shorten_song(filename, path_dataset):
    song = AudioSegment.from_wav(path_dataset + "/" + filename)
    # we trim the song files from 1 to 4 seconds
    start_sec = int(1 * 1000)
    end_sec = int(4 * 1000)
    cut_audio = song[start_sec:end_sec]

    return cut_audio

def shorten_speech(filename, path_dataset):
    speech = AudioSegment.from_wav(path_dataset + "/" + filename)
    # we trim the speech files from 0.25 to 3.25 seconds
    start_sec = int(0.25 * 1000)
    end_sec = int(3.25 * 1000)
    cut_audio = speech[start_sec:end_sec]

    return cut_audio


def split_train_test(path_dataset_trimmed, path_train, path_test):
    all_files =  os.listdir(path_dataset_trimmed)
    all_files_sz = len(all_files)

    # we do a 80%/20% train/test split on the data
    test_sz = math.floor(0.2 * all_files_sz)

    # randomly sample 20% of the dataset to be the test set
    test_data = random.sample(all_files, test_sz)

    train_data = [x for x in all_files if x not in test_data]

    for file in train_data:
        shutil.move(path_dataset_trimmed + "/" + file, path_train)

    for file in test_data:
        shutil.move(path_dataset_trimmed + "/" + file, path_test)

def main():
    path_dataset = "./Dataset"
    # create a temporary directory to hold all the data samples
    os.mkdir(path_dataset)
    get_data_samples(path_dataset)

    path_dataset_trimmed = "./DatasetTrimmed"
    # create a temporary directory to hold all the trimmed data samples
    os.mkdir(path_dataset_trimmed)
    for filename in os.listdir(path_dataset):
        return_file = shorten_audio(filename, path_dataset)
        return_file.export(path_dataset_trimmed + "/" + filename, "wav")

    # remove the temporary Dataset directory
    # directories must be empty to be removed
    for file in os.listdir(path_dataset):
        os.remove(path_dataset + "/" + file)
    os.rmdir(path_dataset)


    path_train = "./Train"
    path_test = "./Test"
    os.mkdir(path_train)
    os.mkdir(path_test)
    split_train_test(path_dataset_trimmed, path_train, path_test)

    # remove the temporary trimmed dataset directory
    os.rmdir(path_dataset_trimmed)

    train_sz = len(os.listdir(path_train))
    test_sz = len(os.listdir(path_test))
    print("Train size:", train_sz)
    print("Test size:", test_sz)
    print("Dataset size:", train_sz + test_sz)

if __name__ == "__main__":
    main()

