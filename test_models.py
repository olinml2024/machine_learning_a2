from pathlib import Path
import numpy as np
import tensorflow as tf

model_file_names = [
    "food_classifier.keras",
]
# model_file_names = [
#     "model1_weights.keras",
#     "model2_weights.keras",
#     "model3_weights.keras",
# ]

# num_correct = 0
# num_total = 0
# predictions_and_labels = []

# for i in range(len(val_data)):
#   predictions = np.argmax(model_ResNet.predict(val_data[i][0]), axis=1)
#   gt_labels = np.argmax(val_data[i][1], axis=1)
#   for i in range(len(predictions)):
#     predictions_and_labels.append([int(predictions[i]), int(gt_labels[i])])
#   num_correct += sum(predictions == gt_labels)
#   num_total += len(predictions)

#   print(f"Accuracy: {num_correct}/{num_total}, {round(num_correct/num_total, 4)*100}%")

# data = {
#     "class_indices": val_data.class_indices,
#     "num_images_by_dessert": num_images_by_dessert,
#     "predictions_and_labels": predictions_and_labels
# }

# with open("results.json", "w") as file:
#   json.dump(data, file)

DATA_DIR = Path(__file__).parent / ""
TEST_DATA_DIR = Path(__file__).parent / "data/test_data"
TEST_IMG_DATA_FP = TEST_DATA_DIR / "test_images.npy"
MODEL_WEIGHTS_HOME_DIR = Path(__file__)


def test_model(model_file_name):
    test_imgs = np.load(TEST_IMG_DATA_FP)
    tf_model = tf.keras.models.load_model()

    predictions = np.argmax(tf_model.predict(test_imgs), axis=1)
    gt_labels = np.array([i//100 for i in range(len(predictions))])
    num_correct = sum(predictions == gt_labels)
    num_total = len(predictions)
    
    print(f"Accuracy: {num_correct}/{num_total}, {round(num_correct/num_total, 4)*100}%")