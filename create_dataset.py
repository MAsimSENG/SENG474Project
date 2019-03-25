import os
from os import listdir
import shutil

count = 0
num_actors = 24

# create a directory to hold all the data samples
path = "./Dataset"
os.mkdir(path)

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
            shutil.move("./Audio_Song_Actors_01-24/Actor_{}/".format(actor) + file, './Dataset')


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
        shutil.move("./Audio_Speech_Actors_01-24/Actor_{}/".format(actor) + file, './Dataset')

print("Number of data samples: ", count)

