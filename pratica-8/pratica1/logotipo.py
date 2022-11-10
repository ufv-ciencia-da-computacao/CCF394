import cv2
import pandas as pd
from glob import glob
import os
from itertools import repeat

def get_files(path):
    files=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            dir = os.path.join(root, name)
            files.extend(list(zip(repeat(name), glob(f"{dir}/*.png"))))
            files.extend(list(zip(repeat(name), glob(f"{dir}/*.bmp"))))
    return files

def get_moments(files):
    momentos=[]
    for classname, file in files:
        print(file)
        image = cv2.imread(file)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh1=255-image

        ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # cv2.imshow("Figura", thresh1)
        # cv2.waitKey(2000)

        moms = cv2.HuMoments(cv2.moments(thresh1)).flatten()
        momentos.append([classname, *moms])
    return momentos

def segment_data():
    # bentley b-15 teve uma segmentação muito ruim
    # acer b-14 e b-41 tiveram uma segmentação muito ruim
    # aerosmith b-107 e b-25 tiveram uma segmação muito ruim
    # air-jordan b-101 e b-108 não seriam bem classificadas
    # apple b-102, b-78, b-8 e b-90 tiveram uma segmentação muito ruim
    # chevrolet b-89 e b-85 tiveram uma segmentação muito ruim
    # linux b-52 teve uma segmentação muito ruim
    # rolling-stones b-95, b-99, b-79, b-35, b-125, b-69, b-119, b-63 e b-90 tiveram uma segmentação muito ruim
    # skype b-19 teve uma segmentação muito ruim
    # b96 los angeles lakers tem 3 logotipos. Excluimos
    
    logotipo1 = get_files("resources/logotipo1")
    logotipo2 = get_files("resources/logotipo2") 

    moments1 = get_moments(logotipo1) # b
    moments1 = pd.DataFrame(moments1, columns=["classname", "m1", "m2", "m3", "m4", "m5", "m6", "m7"])
    moments2 = get_moments(logotipo2)
    moments2 = pd.DataFrame(moments2, columns=["classname", "m1", "m2", "m3", "m4", "m5", "m6", "m7"])

    moments = pd.concat([moments1, moments2], axis=0).reset_index(drop=True)

    def min_max_scaling(column):
        return(column-column.min())/(column.max()-column.min())

    classnames = moments.classname.unique().tolist()
    for col in moments.select_dtypes(exclude=["object"]):
        for x in classnames:
            moments.loc[moments["classname"]==x, col] = min_max_scaling(moments.loc[moments["classname"]==x, col])

    moments.to_csv("moments_logotipo.csv", index=False)

def classify_logotipo():
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn import svm
    import seaborn as sns
    import matplotlib.pyplot as plt

    moments = pd.read_csv("moments_logotipo.csv")

    X = moments.drop("classname", axis=1)
    Y = moments["classname"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 40)

    clf = svm.SVC(kernel="rbf")
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)

    print(f"Acurácia: {accuracy_score(Y_test, y_pred)}")
    print(confusion_matrix(Y_test, y_pred))

    sns.heatmap(confusion_matrix(Y_test, y_pred), annot=True, fmt="d")
    plt.show()

if __name__ == "__main__":
    # segment_data()
    # prepare_data2classify()
    classify_logotipo()



