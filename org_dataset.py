import glob
# from shutil import copyfile
import shutil
import os

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  # Define emotion order
participants = glob.glob("data\\source_emotions\\*")  # Returns a list of all folders with participant numbers

for x in participants:
    part = "%s" % x[-4:]  # store current participant number
    for sessions in glob.glob("%s\\*" % x):  # Store list of sessions for current participant
        for files in glob.glob("%s\\*" % sessions):
            current_session = files[20:-30]
            file = open(files, 'r')

            emotion = int(
                float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.

            print part, current_session
            sourcefile_emotion = glob.glob("data\source_images\\%s\\*" % (current_session))[-1]  # get path for last image in sequence, which contains the emotion
            sourcefile_neutral = glob.glob("data\source_images\\%s\\*" % (current_session))[0]  # do same for neutral image

            dest_neut = "data\sorted_set\\neutral\\%s" % sourcefile_neutral[25:]  # Generate path to put neutral image
            dest_emot = "data\sorted_set\\%s\\%s" % (emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image

            dest_neut2 = dest_neut[:-22]
            dest_emot2 = dest_emot[:-22]
            print dest_neut[:-22]
            print dest_emot[:-22]
            if not os.path.exists(dest_neut2):
                os.mkdir(dest_neut2)
                print "success neutral"

            if not os.path.exists(dest_emot2):
                os.mkdir(dest_emot2)
                print "success emot"

            shutil.copy2(sourcefile_neutral, dest_neut2)
            shutil.copy2(sourcefile_emotion, dest_emot2)