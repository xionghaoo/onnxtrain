from onnxruntime import InferenceSession
from onnxruntime.capi import _pybind_state as C

from IPython.display import Image, display
import numpy as np

# Preprocess the images and convert to tensors as expected by the model
# Makes the image a square and resizes it to 224x224 as is expected by
# the mobilenetv2 model
# Normalize the image by subtracting the mean (0.485, 0.456, 0.406) and
# dividing by the standard deviation (0.229, 0.224, 0.225)
def image_file_to_tensor(file):
    from PIL import Image

    image = Image.open(file)
    width, height = image.size
    if width > height:
        left = (width - height) // 2
        right = (width + height) // 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) // 2
        bottom = (height + width) // 2
    image = image.crop((left, top, right, bottom)).resize((224, 224))

    pix = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
    pix = pix / 255.0
    pix[0] = (pix[0] - 0.485) / 0.229
    pix[1] = (pix[1] - 0.456) / 0.224
    pix[2] = (pix[2] - 0.406) / 0.225
    return pix

# Training metadata
dog, cat, elephant, cow, airpod = "dog", "cat", "elephant", "cow", "airpod" # labels
label_to_id_map = {
    "dog": 0,
    "cat": 1,
    "elephant": 2,
    "cow": 3,
    "airpod": 4
} # label to index mapping

num_samples_per_class = 20
num_epochs = 5


# ort training api - export the model for so that it can be used for inferencing
# model.export_model_for_inferencing("inference_artifacts/inference.onnx", ["output"])

# Run inference on the exported model
session = InferenceSession("inference_artifacts/inference.onnx", providers=C.get_available_providers())

def softmax(logits):
    return (np.exp(logits)/np.exp(logits).sum())

def predict(test_file, test_name):
    logits = session.run(["output"], {"input": np.stack([image_file_to_tensor(test_file)])})
    probabilities = softmax(logits) * 100
    display(Image(filename=test_file))
    print_prediction(probabilities, test_name)

def print_prediction(prediction, test_name):
    print(f"test\t{dog}\t{cat}\t{elephant}\t{cow}\t{airpod}")
    print("-------------------------------------------------")
    print(f"{test_name}\t{prediction[0][0][0]:.2f}\t{prediction[0][0][1]:.2f}\t{prediction[0][0][2]:.2f}\t\t{prediction[0][0][3]:.2f}\t\t{prediction[0][0][4]:.2f}")


# Test on sample image (test1.jpg)
# predict("inference_artifacts/test3.jpeg", "test3")
predict("inference_artifacts/test4.png", "test4")