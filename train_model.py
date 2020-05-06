"""
This script will train a model on the RaspberryPI to execute later.

The reason we have to train the RPI, is because you cannot train on one CPU architecture and transfer the
saved model and have it run on a different CPU architecture.

The parameters of the model are determined outside the RPI, and this script just trains the best LogisticRegression parameters.

"""
from sklearn.linear_model import LogisticRegression
from imutils import paths
import cv2
from joblib import dump

# 0, 1, 2
directions = ["left", "straight", "right"]

image_path_root = "/home/pi/dev/training_data"


def get_image_data():
    imagePaths = list(paths.list_images(image_path_root))
    direction_vector = []
    images_vector = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        # images should be gray scale but make sure
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        print(f"Shape: {image.shape}, Image: {imagePath}")
        flatten_image = image.flatten()
        print(f"Flat Shape: {flatten_image.shape}")
        direction = directions.index(imagePath.split("/")[-2])
        images_vector.append(flatten_image)
        direction_vector.append(direction)

    return images_vector, direction_vector


def get_model():
    """
    Another project, using the image training data, determined the best LogisticRegression parameters.

    The reason we cannot do this on the RPI is because it would take to long to RandomGridSearch on the RPI.
    """
    clf = LogisticRegression(penalty="l2", C=0.0001, solver='saga', multi_class='auto')

    # KNN was determined by TPOT
    # this turned out to be too slow.  each predict was about 0.2 seconds
    # clf = KNeighborsClassifier(n_neighbors=93, p=1, weights="uniform")
    return clf


def train_save_model(model, X, y):
    model.fit(X, y)
    dump(model, "rpi_gpg3_line_follower_model.sav")
    print("Model trained and saved.")


if __name__ == '__main__':
    model = get_model()
    X, y = get_image_data()
    train_save_model(model, X, y)
