from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
import matplotlib.pyplot as plt


TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_RESULTS_DIR = Path(__file__).parent / "results"
MODEL_WEIGHTS_DIR = Path(__file__).parent.parent / "model_weights"

CLASS_INDICES = {
    "apple_pie": 0,
    "baklava": 1,
    "beignets": 2,
    "bread_pudding": 3,
    "cannoli": 4,
    "carrot_cake": 5,
    "cheesecake": 6,
    "chocolate_cake": 7,
    "chocolate_mousse": 8,
    "churros": 9,
    "creme_brulee": 10,
    "croque_madame": 11,
    "cup_cakes": 12,
    "donuts": 13,
    "french_toast": 14,
    "frozen_yogurt": 15,
    "ice_cream": 16,
    "macarons": 17,
    "pancakes": 18,
    "red_velvet_cake": 19,
    "strawberry_shortcake": 20,
    "tiramisu": 21,
    "waffles": 22,
}
CLASS_INDICES_REVERSED = {value: key for (key, value) in CLASS_INDICES.items()}
MODEL_NAMES = ["model0", "model1", "model2"]
NUM_IMAGES_PER_CLASS = 100


def test_model(model_name):
    """
    Based on the inputted model name, this class loads the respective weights,
    along with the full testing data (This can be compressed to only use half
    the data if RAM is of concern while running the process locally). The testing
    data is then extracted and saves as a json in `~/model_testing/results/{model_name}.json`.
    """

    # Load in the testing datasets.
    num_images = NUM_IMAGES_PER_CLASS * len(CLASS_INDICES.keys())
    test_imgs = np.zeros([num_images, 250, 250, 3])

    # The RGB values of each image need to be normalize (placed on a scale 0-1
    # in order to be compatible with the model)
    test_imgs[: num_images // 2, :, :, :] = (
        np.load(TEST_DATA_DIR / "testing_data_1.npy") / 255
    )
    test_imgs[num_images // 2 :, :, :, :] = (
        np.load(TEST_DATA_DIR / "testing_data_2.npy") / 255
    )

    # Load in the specific TF model that was specified.
    ResNet_V2_50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"

    food_classifier = tf.keras.Sequential(
        [
            hub.KerasLayer(
                ResNet_V2_50,
                trainable=False,
                input_shape=(250, 250, 3),
                name="Resnet_V2_50",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(23, activation="softmax", name="Output_layer"),
        ]
    )
    food_classifier.load_weights(MODEL_WEIGHTS_DIR / f"{model_name}.ckpt")

    # Load the GT labels
    with open(TEST_DATA_DIR / "testing_labels.json", "r") as file:
        gt_label_mappings = json.load(file)["labels"]
    gt_labels = [
        CLASS_INDICES[gt_label_mappings[str(idx)]] for idx in range(num_images)
    ]

    # Extract predictions from the TF model on the test data
    predictions = np.argmax(food_classifier.predict(test_imgs), axis=1)

    # Calculate results and error metrics for the model's performance
    num_correct = sum(predictions == gt_labels)
    num_total = len(predictions)
    print(
        f"Accuracy: {num_correct}/{num_total}, {round(num_correct/num_total, 4)*100}%"
    )

    predictions_and_labels = []
    for idx in range(predictions.shape[0]):
        predictions_and_labels.append(
            {
                "prediction_label": CLASS_INDICES_REVERSED[predictions[idx]],
                "prediction_idx": int(predictions[idx]),
                "gt_label": CLASS_INDICES_REVERSED[gt_labels[idx]],
                "gt_idx": gt_labels[idx],
            }
        )

    # Store the results in a JSON for further processing by the student.
    results = {
        "class_indices": CLASS_INDICES,
        "predictions_and_labels": predictions_and_labels,
    }

    with open(TEST_RESULTS_DIR / f"{model_name}.json", "w") as file:
        json.dump(results, file, indent=4)


def test_model_on_image(model_name, path_to_image: Path, visualize=False):
    """
    This is a helper function provided for students who wish to run their own
    testing on external images.

    Args:
        model_name: (model0, model1, model2) The name of the model that should be loaded
        path_to_image: The full path to the image that should be tested
        visualize: Boolean defining if the reshaped image should be displayed
    """

    # Load the desired model
    ResNet_V2_50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5"

    food_classifier = tf.keras.Sequential(
        [
            hub.KerasLayer(
                ResNet_V2_50,
                trainable=False,
                input_shape=(250, 250, 3),
                name="Resnet_V2_50",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(23, activation="softmax", name="Output_layer"),
        ]
    )
    food_classifier.load_weights(MODEL_WEIGHTS_DIR / f"{model_name}.ckpt")

    # Load the image and normalize the RBG values to be [0, 1]
    img = np.zeros([1, 250, 250, 3], dtype=float)
    img[0] = np.asarray(Image.open(path_to_image).resize([250, 250])) / 255

    if visualize:
        plt.imshow(img[0])
        plt.show()

    # Run the forward pass of the TF Model to generate the prediction
    prediction = np.argmax(food_classifier.predict(img))
    food_label = CLASS_INDICES_REVERSED[prediction]

    # Return the prediction idx and the human-readable label
    return prediction, food_label


if __name__ == "__main__":
    for model in MODEL_NAMES:
        test_model(model)
