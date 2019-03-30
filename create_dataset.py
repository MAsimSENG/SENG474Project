import os
import shutil
import math
import numpy as np

def get_data_samples(path_dataset):
    '''
    This function retrieves the relevant wav files (i.e.) emotion label one of:
    03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    and places the wav files in the directory specified as input

    :param path_dataset: str: the path to the directory where the relevant wav files will be placed in
    '''

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
            files_song = os.listdir("./Audio_Song_Actors_01-24/Actor_{}".format(actor))
            # id of wav file extracted from the wav file's title
            files_song_id = [(f,f.rstrip(".wav").split("-")) for f in files_song]

            for file,id in files_song_id:
                # the relevant emotions correspond to emotion labels in [3,4,5,6,7,8]
                if int(id[2]) in [3, 4, 5, 6, 7, 8]:
                    wav_files_song.append(file)

            for file in wav_files_song:
                shutil.move("./Audio_Song_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)

        files_speech = os.listdir("./Audio_Speech_Actors_01-24/Actor_{}".format(actor))
        files_speech_id = [(f,f.rstrip(".wav").split("-")) for f in files_speech]

        for file,id in files_speech_id:
            if int(id[2]) in [3,4,5,6,7,8]:
                wav_files_speech.append(file)

        # move files to Dataset directory
        for file in wav_files_speech:
            shutil.move("./Audio_Speech_Actors_01-24/Actor_{}/".format(actor) + file, path_dataset)



def split_train_test(path_dataset, path_train, path_test):
    '''
    This function splits the wav files into a train folder and test folder. We split the data
    into an 80/20 train/test split, constrained so that exactly 20% of each emotion is in the
    test folder.
    :param path_dataset: str: path to the relevant wav files
    :param path_train: str: path to the train directory
    :param path_test: str: path to the test directory
    '''

    all_files =  os.listdir(path_dataset)
    train_files = []
    test_files = []

    # separate the files according to their emotion label
    emotion_dict = {3:[],4:[],5:[],6:[],7:[],8:[]}
    for file in all_files:
        # append file to the list in emotion_dict corresponding to the file's emotion label
        emotion_label = int(file.rstrip(".wav").split("-")[2])
        emotion_dict[emotion_label].append(file)

    #for each emotion label, we split the files into an 80/20 train/test split
    for i in range(3,9):
        perm = np.random.permutation(len(emotion_dict[i]))
        test_sz = math.floor(0.2 * len(emotion_dict[i]))
        # according to the random permutation (to achieve a random split), take the first 20% of the files
        # for the particular emotion and add to list of test files
        test_files += [emotion_dict[i][perm[j]] for j in range(test_sz)]
        # remaining 80% of the files appended to set of train files
        train_files += [emotion_dict[i][perm[j]] for j in range(test_sz, len(emotion_dict[i]))]

    for file in train_files:
        if file in test_files:
            a = 1
            # Could raise error here

    # move train and test files into their corresponding directories
    for file in train_files:
        shutil.move(path_dataset + "/" + file, path_train)

    for file in test_files:
        shutil.move(path_dataset + "/" + file, path_test)

def main():
    path_dataset = "./Dataset"
    # create a temporary directory to hold all the data samples
    os.mkdir(path_dataset)
    get_data_samples(path_dataset)

    path_train = "./Train"
    path_test = "./Test"
    os.mkdir(path_train)
    os.mkdir(path_test)
    split_train_test(path_dataset, path_train, path_test)

    # remove the temporary dataset directory
    os.rmdir(path_dataset)

    train_sz = len(os.listdir(path_train))
    test_sz = len(os.listdir(path_test))
    print("Train size:", train_sz)
    print("Test size:", test_sz)
    print("Dataset size:", train_sz + test_sz)

    test_dis = np.zeros(6)
    train_dis = np.zeros(6)

    for file in os.listdir(path_train):
        emotion = int(file.rstrip(".wav").split("-")[2])
        train_dis[emotion-3] += 1

    for file in os.listdir(path_test):
        emotion = int(file.rstrip(".wav").split("-")[2])
        test_dis[emotion-3] += 1

    # check that 20% of the files for each emotion are placed in the test set
    print("Test distribution:", test_dis / (train_dis + test_dis))

if __name__ == "__main__":
    main()

