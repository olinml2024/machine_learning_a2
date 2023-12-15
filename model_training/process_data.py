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
    # if os.path.isdir(testing_images_dir):
    #     shutil.rmtree(testing_images_dir)

    # for file in os.walk(Path(__file__).parent / "images"):
    #     if "images/" in file[0]:
    #         folder_name = file[0].split("/")[-1]
    #         if folder_name in desserts:
    #             os.makedirs(f"{testing_images_dir}/{folder_name}", exist_ok=True)
    #             curr_img = 0
    #             for filename in file[2]:
    #                 shutil.move(
    #                     f"{file[0]}/{filename}",
    #                     f"{testing_images_dir}/{folder_name}/{filename}",
    #                 )
    #                 curr_img += 1
    #                 if curr_img >= NUM_TEST_IMAGES:
    #                     break

    # Store images
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

    # Testing data needs to be split into 2 because it exceeds the 2.5 GB
    # limit for GitHub LFS
    testing_data_1 = testing_data[: (NUM_TEST_IMAGES * len(desserts)) // 2, :, :, :]
    testing_data_2 = testing_data[(NUM_TEST_IMAGES * len(desserts)) // 2 :, :, :, :]

    with open(testing_data_dir / "testing_labels.json", "w") as file:
        json.dump(testing_labels, file, indent=4)

    np.save(testing_data_dir / "testing_data_1.npy", testing_data_1)
    np.save(testing_data_dir / "testing_data_2.npy", testing_data_2)


def generate_train_data():
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
        if os.path.isdir(training_dir):
            shutil.rmtree(training_dir)

        for file in os.walk(Path(__file__).parent / "images"):
            if "images/" in file[0]:
                folder_name = file[0].split("/")[-1]
                if folder_name in desserts:
                    os.makedirs(f"{training_dir}/{folder_name}", exist_ok=True)
                    curr_img = 0
                    for filename in file[2]:
                        shutil.copy(
                            f"{file[0]}/{filename}",
                            f"{training_dir}/{folder_name}/{filename}",
                        )
                        curr_img += 1
                        if curr_img >= num_images_by_dessert[folder_name]:
                            break


def train_model():
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
        tf.keras.utils.plot_model(model_ResNet)
        model_ResNet.fit(train_data, epochs=10, verbose=1)
        model_ResNet.save(Path(__file__).parent / f"model_weights/{model_name}.keras")


if __name__ == "__main__":
    generate_test_data()
    # generate_train_data()
    # train_model()
    pass
