"""
This file is used to generate training and test image datasets based on the
kaggle food classification competition (https://www.kaggle.com/datasets/kmader/food41).
As a requirement, this dataset must be extracted in the directory parallel to this file
(ie. ~/images/*). If you would like to place these images in a different directory, update
`kaggle_dataset_dir` accordingly to point to that location.
"""

import numpy as np  # linear algebra
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from PIL import Image

# import cv2
import json
import shutil
import os
from pathlib import Path

NUM_TEST_IMAGES = 100

kaggle_dataset_dir = Path(__file__).parent / "images"
testing_images_dir = Path(__file__).parent.parent / "model_testing/test_images"
testing_data_dir = Path(__file__).parent.parent / "model_testing/test_data"
base_training_dir = Path(__file__).parent / "train_images"

desserts = [
    "apple_pie",
    "baklava",
    "beignets",
    "bread_pudding",
    "cannoli",
    "carrot_cake",
    "cheesecake",
    "chocolate_cake",
    "chocolate_mousse",
    "churros",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "donuts",
    "french_toast",
    "frozen_yogurt",
    "ice_cream",
    "macarons",
    "pancakes",
    "red_velvet_cake",
    "strawberry_shortcake",
    "tiramisu",
    "waffles",
]


def generate_test_data():
    """
    This function extracts test images from `kaggle_dataset_dir` and moves them
    into their respective folders in `testing_images_dir`. There will be
    `NUM_IMAGES` of each dessert class.
    """
    for file in os.walk(kaggle_dataset_dir):
        if "images/" in file[0]:
            folder_name = file[0].split("/")[-1]
            if folder_name in desserts:
                os.makedirs(f"{testing_images_dir}/{folder_name}", exist_ok=True)
                curr_img = 0
                # Move files into the testing dataset directory until enough
                # images have been selected
                for filename in file[2]:
                    shutil.move(
                        f"{file[0]}/{filename}",
                        f"{testing_images_dir}/{folder_name}/{filename}",
                    )
                    curr_img += 1
                    if curr_img >= NUM_TEST_IMAGES:
                        break

    # In addition to storing the images, we store numpy representations of the
    # testing dataset for ease of passing each image into the trained tensorflow
    # models.
    num = 0
    testing_data = np.zeros([NUM_TEST_IMAGES * len(desserts), 250, 250, 3])
    testing_labels = {"labels": {}}

    for file in os.walk(testing_images_dir):
        if file[0].startswith(f"{testing_images_dir}/"):
            folder_name = file[0].split("/")[-1]
            if folder_name in desserts:
                for filename in file[2]:
                    img = Image.open(f"{testing_images_dir}/{folder_name}/{filename}")
                    img = img.resize([250, 250])
                    testing_data[num] = np.asarray(img.resize([250, 250]))
                    testing_labels["labels"][num] = folder_name
                    num += 1
                    print(f"{num}/{NUM_TEST_IMAGES*len(desserts)}")

    # Testing data needs to be split into 2 due to intermittent issues loading the
    # size of each file into RAM during testing.
    testing_data_1 = testing_data[: (NUM_TEST_IMAGES * len(desserts)) // 2, :, :, :]
    testing_data_2 = testing_data[(NUM_TEST_IMAGES * len(desserts)) // 2 :, :, :, :]

    # This JSON represents the ground truth label of each testing image, mapping
    # the idx as they appear in the npy files with the corresponding label.
    with open(testing_data_dir / "testing_labels.json", "w") as file:
        json.dump(testing_labels, file, indent=4)

    np.save(testing_data_dir / "testing_data_1.npy", testing_data_1)
    np.save(testing_data_dir / "testing_data_2.npy", testing_data_2)


def generate_train_data():
    """
    This function extracts training images for each respective model. To artificially
    inject diversity between the performance of each model, they are trained on a
    different number of images per dessert class, and have no overlapping images
    between eachother.
    """

    # Three different schemes for determining the number of images per class for
    # each model.
    num_images_by_model = [
        [int(x) for x in np.random.choice(range(100, 500), len(desserts))],
        [int(x) for x in np.random.choice([200, 400], len(desserts))],
        [int(x) for x in np.random.choice([300], len(desserts))],
    ]

    for model_idx, num_images in enumerate(num_images_by_model):
        num_images_by_dessert = {
            desserts[i]: num_images[i] for i in range(len(desserts))
        }

        training_dir = base_training_dir / f"model{model_idx}"

        for file in os.walk(kaggle_dataset_dir):
            if "images/" in file[0]:
                folder_name = file[0].split("/")[-1]
                if folder_name in desserts:
                    os.makedirs(f"{training_dir}/{folder_name}", exist_ok=True)
                    curr_img = 0
                    for filename in file[2]:
                        shutil.move(
                            f"{file[0]}/{filename}",
                            f"{training_dir}/{folder_name}/{filename}",
                        )
                        curr_img += 1
                        if curr_img >= num_images_by_dessert[folder_name]:
                            break


def train_model():
    """
    This function is used to train each tensorflow model based on their
    respective training data. For convenience, this functionality has
    been copied into `train_food_classifiers.ipynb` to be compatible
    with Google Colab.
    """
    for model_name in ["model0", "model1", "model2"]:
        training_dir = base_training_dir / model_name
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.0)

        train_data = train_datagen.flow_from_directory(
            training_dir,
            target_size=(250, 250),
            batch_size=32,
            class_mode="categorical",
            subset="training",
        )
        ResNet_V2_50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"

        model_ResNet = tf.keras.Sequential(
            [
                hub.KerasLayer(
                    ResNet_V2_50,
                    trainable=False,
                    input_shape=(250, 250, 3),
                    name="Resnet_V2_50",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    len(desserts), activation="softmax", name="Output_layer"
                ),
            ]
        )

        model_ResNet.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        model_ResNet.summary()
        model_ResNet.fit(train_data, epochs=10, verbose=1)
        model_ResNet.save(Path(__file__).parent / f"model_weights/{model_name}.keras")


if __name__ == "__main__":
    generate_test_data()
    generate_train_data()
    train_model()
