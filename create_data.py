import HoG as HoG
import numpy as np
import os

# function to create data
def create_data():
    files = os.listdir('./positive_train')
    for i in range(len(files)):
        hog_obj = HoG.HOG('./positive_train/'+files[i])
        hog = hog_obj.hog()
        if i == 0:
            train_X_1 = hog
        else:
            train_X_1 = np.vstack((train_X_1, hog))

    train_y_1 = np.full((train_X_1.shape[0],1), 1)

    files = os.listdir('./negative_train')
    for i in range(len(files)):
        hog_obj = HoG.HOG('./negative_train/'+files[i])
        hog = hog_obj.hog()
        if i == 0:
            train_X_0 = hog
        else:
            train_X_0 = np.vstack((train_X_0, hog))

    train_y_0 = np.full((train_X_0.shape[0],1), 0)

    train_x = np.vstack((train_X_1, train_X_0))
    train_x = np.nan_to_num(train_x)
    train_y = np.vstack((train_y_1, train_y_0))
    train = np.append(train_x, train_y, 1)

    # saving training data to csv file
    np.savetxt("train.csv",train,delimiter=",")

    files = os.listdir('./positive_test')
    for i in range(len(files)):
        hog_obj = HoG.HOG('./positive_test/'+files[i])
        hog = hog_obj.hog()
        if i == 0:
            test_X_1 = hog
        else:
            test_X_1 = np.vstack((test_X_1, hog))

    test_y_1 = np.full((test_X_1.shape[0],1), 1)

    files = os.listdir('./negative_test')
    for i in range(len(files)):
        hog_obj = HoG.HOG('./negative_test/'+files[i])
        hog = hog_obj.hog()
        if i == 0:
            test_X_0 = hog
        else:
            test_X_0 = np.vstack((test_X_0, hog))

    test_y_0 = np.full((test_X_0.shape[0],1), 0)

    test_x = np.vstack((test_X_1, test_X_0))
    test_x = np.nan_to_num(test_x)
    test_y = np.vstack((test_y_1, test_y_0))
    test = np.append(test_x, test_y, 1)

    # saving testing data to csv file
    np.savetxt("test.csv",test,delimiter=",")

create_data()
