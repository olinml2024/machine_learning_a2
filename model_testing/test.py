from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json

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
    num_images = NUM_IMAGES_PER_CLASS * len(CLASS_INDICES.keys())
    test_imgs = np.zeros([num_images, 250, 250, 3])
    test_imgs[: num_images // 2, :, :, :] = (
        np.load(TEST_DATA_DIR / "testing_data_1.npy") / 255
    )
    test_imgs[num_images // 2 :, :, :, :] = (
        np.load(TEST_DATA_DIR / "testing_data_2.npy") / 255
    )
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

    with open(TEST_DATA_DIR / "testing_labels.json", "r") as file:
        gt_label_mappings = json.load(file)["labels"]
    gt_labels = [
        CLASS_INDICES[gt_label_mappings[str(idx)]] for idx in range(num_images)
    ]

    predictions = np.argmax(food_classifier.predict(test_imgs), axis=1)
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

    results = {
        "class_indices": CLASS_INDICES,
        "predictions_and_labels": predictions_and_labels,
    }

    with open(TEST_RESULTS_DIR / f"{model_name}.json", "w") as file:
        json.dump(results, file, indent=4)


if __name__ == "__main__":
    for model in MODEL_NAMES:
        test_model(model)
