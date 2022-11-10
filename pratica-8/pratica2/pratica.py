import cv2
import pandas as pd
from glob import glob
import os
from itertools import repeat
import mahotas as mt

def get_files(path):
    files=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            dir = os.path.join(root, name)
            files.extend(list(zip(repeat(name), glob(f"{dir}/*.png"))))
    return files

def get_moments(files, zernike=False, radius=21):
    momentos=[]
    for classname, file in files:
        # print(file)
        image = cv2.imread(file)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh1=255-image

        ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # cv2.imshow("Figura", thresh1)
        # cv2.waitKey(2000)
        if not zernike:
            moms = cv2.HuMoments(cv2.moments(thresh1)).flatten()
            momentos.append([classname, *moms])
        else:
            moms = mt.features.zernike_moments(thresh1, radius)
            momentos.append([classname, *moms])

    return momentos

def segment_data(zernike=False, radius=7):   
    gestos = get_files("dataset_full/")
    gestos = get_moments(gestos, zernike, radius)
    # print(gestos)

    moments = pd.DataFrame(gestos, columns=["classname"] + [f"m{i}" for i in range(1, len(gestos[0]))])

    def min_max_scaling(column):
        return(column-column.min())/(column.max()-column.min())

    classnames = moments.classname.unique().tolist()
    for col in moments.select_dtypes(exclude=["object"]):
        for x in classnames:
            moments.loc[moments["classname"]==x, col] = min_max_scaling(moments.loc[moments["classname"]==x, col])

    if zernike:
        moments.to_csv("zernike.csv", index=False)
    else:
        moments.to_csv("hu_moments_gestos.csv", index=False)

def classify_gestos(dataset, pca=False):
    from sklearn.model_selection import train_test_split
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn import svm
    import seaborn as sns
    import matplotlib.pyplot as plt

    moments = pd.read_csv(dataset)

    if pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.95)
        moments_pca = pca.fit_transform(moments.drop("classname", axis=1))
        print("PCA Components: " + pca.n_components_)
        moments_pca = pd.DataFrame(moments_pca, columns=[f"p{i}" for i in range(1, pca.n_components_+1)])
        moments_pca["classname"] = moments.classname
        moments = moments_pca.copy()

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
    ### Hu moments -> Acurácia: 0.9841777624070551
    # segment_data()
    # classify_gestos("hu_moments_gestos.csv")


    ### Zernike Moments -> Acurácia: 0.9741483659000518
    # segment_data(zernike=True, radius=21)
    # classify_gestos("zernike.csv")

    ### PCA + Zernike Moments -> Acurácia: 0.9757911118796473
    classify_gestos("zernike.csv", pca=True)





