import os
import shutil
import math
import numpy as np

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
                if int(id[2]) in [3, 4, 5, 6, 7, 8]:
                    wav_files_song.append(file)
                    count += 1

            # move files to Dataset directory
            for file in wav_files_song:
                shutil.move("./Audio_Song_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)


        # files in the actor's directory for speech directory
        files_speech = os.listdir("./Audio_Speech_Actors_01-24/Actor_{}".format(actor))
        files_speech_id = [(f,f.rstrip(".wav").split("-")) for f in files_speech]

        for file,id in files_speech_id:
            if int(id[2]) in [3,4,5,6,7,8]:
                wav_files_speech.append(file)
                count += 1

        # move files to Dataset directory
        for file in wav_files_speech:
            shutil.move("./Audio_Speech_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)



def split_train_test(path_dataset, path_train, path_test):
    all_files =  os.listdir(path_dataset)
    # list of files names to put in train folder
    train_files = []
    # list of file names to put in test folder
    test_files = []

    # separate the files according to their emotion label
    emotion_dict = {3:[],4:[],5:[],6:[],7:[],8:[]}
    for file in all_files:
        emotion_label = int(file.rstrip(".wav").split("-")[2])
        emotion_dict[emotion_label].append(file)

    #for each emotion, we split the files into an 80/20 train/test split
    for i in range(3,9):
        perm = np.random.permutation(len(emotion_dict[i]))
        test_sz = math.floor(0.2 * len(emotion_dict[i]))
        test_files += [emotion_dict[i][perm[j]] for j in range(test_sz)]
        train_files += [emotion_dict[i][perm[j]] for j in range(test_sz, len(emotion_dict[i]))]

    print(train_files)
    print(test_files)

    for file in train_files:
        if file in test_files:
            print("AHHH")

    for file in train_files:
        shutil.move(path_dataset + "/" + file, path_train)

    for file in test_files:
        shutil.move(path_dataset + "/" + file, path_test)

def main():
    path_dataset = "./Dataset"
    # create a temporary directory to hold all the data samples
    os.mkdir(path_dataset)
    class_sample_num = get_data_samples(path_dataset)

    path_train = "./Train"
    path_test = "./Test"
    os.mkdir(path_train)
    os.mkdir(path_test)
    split_train_test(path_dataset, path_train, path_test)

    # remove the temporary trimmed dataset directory
    os.rmdir(path_dataset)

    train_sz = len(os.listdir(path_train))
    test_sz = len(os.listdir(path_test))
    print("Train size:", train_sz)
    print("Test size:", test_sz)
    print("Dataset size:", train_sz + test_sz)
    print("Dataset split:", class_sample_num)

    test_dis = [0,0,0,0,0,0]
    train_dis = [0,0,0,0,0,0]

    for file in os.listdir(path_train):
        emotion = int(file.rstrip(".wav").split("-")[2])
        train_dis[emotion-3] += 1

    for file in os.listdir(path_test):
        emotion = int(file.rstrip(".wav").split("-")[2])
        test_dis[emotion-3] += 1

    print("Train distribution:", train_dis)
    print("Test distribution:", test_dis)

if __name__ == "__main__":
    main()

