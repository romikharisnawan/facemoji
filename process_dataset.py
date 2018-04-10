"""
This file is responsible for harvesting CK database for images of emotions. It gets a neutral face and a emotion face for each subject.
Based on Paul van Gent's code from blog post: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
"""
import glob
import os
# from shutil import copyfile
import shutil

import cv2

from face_detect import find_faces

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

def remove_old_set(emotions):
    """
    Removes old images produced from dataset.
    :param emotions: List of emotions names.
    """
    print("Removing old dataset")
    for emotion in emotions:
        filelist = glob.glob(r"data\sorted_set\%s\*" % emotion)
        for f in filelist:
            os.remove(f)


def harvest_dataset(emotions):
    """
    Copies photos of participants in sessions to new folder.
    :param emotions: List of emotions names.
    """
    print("Harvesting dataset")

    emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness",
                "surprise"]  # Define emotion order
    participants = glob.glob("data\\source_emotions\\*")  # Returns a list of all folders with participant numbers
    print participants
    for x in participants:
        part = "%s" % x[-4:]  # store current participant number
        for sessions in glob.glob("%s\\*" % x):  # Store list of sessions for current participant
            print sessions
            for files in glob.glob("%s\\*" % sessions):
                print files
                current_session = files[20:-30]
                file = open(files, 'r')

                emotion = int(float(file.readline()))  # emotions are encoded as a float, readline as float, then convert to integer.
                print part, current_session

                sourcefile_emotion = glob.glob("data\source_images\\%s\\*" % (current_session))[-1]  # get path for last image in sequence, which contains the emotion
                sourcefile_neutral = glob.glob("data\source_images\\%s\\*" % (current_session))[0]  # do same for neutral image
                print sourcefile_emotion
                print sourcefile_neutral
                path_target = sourcefile_neutral[25:][:4]
                # path_target2 = path_target[:4]
                print path_target
                #
                # sourcefile_emotion = glob.glob("data\source_images\\%s\\%s\\*" % (part, current_session))[-1]  # get path for last image in sequence, which contains the emotion
                # sourcefile_neutral = glob.glob("data\source_images\\%s\\%s\\*" % (part, current_session))[0]  # do same for neutral image

                dest_neut = r'C:\Users\Dell\Documents\facemoji\data\sorted_set\neutral\%s' % sourcefile_neutral[25:]  # Generate path to put neutral image
                dest_emot = r'C:\Users\Dell\Documents\facemoji\data\sorted_set\%s\%s' % (emotions[emotion], sourcefile_emotion[25:])  # Do same for emotion containing image
                dest_emot1 = dest_emot[:-25]
                dest_emot2 = dest_emot[:-22]
                print dest_emot2

                print dest_neut[:60]
                if not os.path.exists(dest_neut[:60]):
                    os.mkdir(dest_neut[:60])

                if not os.path.exists(dest_emot1):
                    os.mkdir(dest_emot1)

                if not os.path.exists(dest_emot2):
                    os.mkdir(dest_emot2)

                shutil.copy2(sourcefile_neutral, dest_neut)
                shutil.copy2(sourcefile_emotion, dest_emot2)
                # copyfile(sourcefile_neutral, dest_neut)  # Copy file
                # copyfile(sourcefile_emotion, dest_emot)  # Copy file



    # participants = glob.glob('data/source_emotions/*')  # returns a list of all folders with participant numbers
    #
    # for participant in participants:
    #     neutral_added = False
    #
    #     for sessions in glob.glob("%s/*" % participant):  # store list of sessions for current participant
    #         print participant
    #         for files in glob.glob("%s/*" % sessions):
    #             current_session = files[20:-30]
    #             file = open(files, 'r')
    #
    #             # emotions are encoded as a float, readline as float, then convert to integer
    #             print file
    #             print file.readline()
    #             emotion = int(float(file.readline()))
    #             images = glob.glob("data/source_images/%s/*" % current_session)
    #
    #             # get path for last image in sequence, which contains the emotion
    #             source_filename = images[-1].split('/')[-1]
    #             # do same for emotion containing image
    #             destination_filename = "data/sorted_set/%s/%s" % (emotions[emotion], source_filename)
    #             # copy file
    #             copyfile("data/source_images/%s/%s" % (current_session, source_filename), destination_filename)
    #
    #             if not neutral_added:
    #                 # do same for neutral image
    #                 source_filename = images[0].split('/')[-1]
    #                 # generate path to put neutral image
    #                 destination_filename = "data/sorted_set/neutral/%s" % source_filename
    #                 # copy file
    #                 copyfile("data/source_images/%s/%s" % (current_session, source_filename), destination_filename)
    #                 neutral_added = True


def extract_faces(emotions):
    """
    Crops faces in emotions images.
    :param emotions: List of emotions names.
    """
    print("Extracting faces")
    for emotion in emotions:
        files = glob.glob('data/sorted_set/%s/*' % emotion)

        filenumber = 0
        for f in files:
            # frame = cv2.imread(f)  # Open image
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
            gray = cv2.imread(f, 0)

            # Detect face using 4 different classifiers
            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
            face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
            face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

            # Go over detected faces, stop at first detected face, return empty if no face.
            if len(face) == 1:
                facefeatures = face
            elif len(face_two) == 1:
                facefeatures = face_two
            elif len(face_three) == 1:
                facefeatures = face_three
            elif len(face_four) == 1:
                facefeatures = face_four
            else:
                facefeatures = ""

            # Cut and save face
            for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
                print "face found in file: %s" % f
                gray = gray[y:y + h, x:x + w]  # Cut the frame to size

                try:
                    out = cv2.resize(gray, (350, 350))  # Resize face so all images have same size
                    cv2.imwrite("data/sorted_set/%s/%s.png" % (emotion, filenumber), out)  # Write image
                except:
                    pass  # If error, pass file
            filenumber += 1  # Increment image number

        # for file_number, photo in enumerate(photos):
        #     frame = cv2.imread(photo)
        #     normalized_faces = find_faces(frame)
        #     # os.remove(photo)
        #
        #     for face in normalized_faces:
        #         try:
        #             cv2.imwrite("data/sorted_set/%s/%s.png" % (emotion, file_number + 1), face[0])  # write image
        #         except:
        #             print("error in processing %s" % photo)


if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    remove_old_set(emotions)
    harvest_dataset(emotions)
    extract_faces(emotions)
