import torch
from torchvision import datasets, models, transforms
import gdown
import itertools
import gc


def load_models():
    # now we will load each model
    gdown.download(id="1MQT3EyNmKPsLXRX8r39x8netudx2kBFt", output="model0.pth")
    model0 = models.resnet18(weights="IMAGENET1K_V1")
    model0.fc = torch.nn.Linear(model0.fc.in_features, 23)
    model0.load_state_dict(torch.load("model0.pth", map_location=torch.device('cpu')))

    gdown.download(id="1BvyuAafFiuiwR4vFRJOCgzeVJTpwsLMq", output="model1.pth")
    model1 = models.resnet50(weights="IMAGENET1K_V1")
    model1.fc = torch.nn.Linear(model1.fc.in_features, 23)
    model1.load_state_dict(torch.load("model1.pth", map_location=torch.device('cpu')))

    gdown.download(id="1Yzpmg0WIQInrHiUuQwJVaw9mjFDBp6JK", output="model2.pth")
    model2 = models.efficientnet_v2_l(weights="IMAGENET1K_V1")
    model2.classifier[1] = torch.nn.Linear(1280, 23)
    model2.load_state_dict(torch.load("model2.pth", map_location=torch.device('cpu')))
    return [model0, model1, model2]

def get_model_preds(model, transforms):
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_set = datasets.ImageFolder('test_data/images', transforms)
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

    model = model.to(device)
    model.eval()

    num_test = len(test_set)
    batch_preds = []
    batch_labels = []
    with torch.no_grad():
        total_processed = 0
        for inputs, labels in test_set_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().detach().numpy()
            batch_labels.append(labels.numpy())
            batch_preds.append(preds)
            total_processed += preds.shape[0]
            if total_processed % 96 == 0:
                print(f"{total_processed}/{num_test}")
    model_ground_truth_labels = list(itertools.chain(*batch_labels))
    model_preds = list(itertools.chain(*batch_preds))
    return model_ground_truth_labels, model_preds

def test_on_image(model, image):
    no_alterations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        as_tensor = no_alterations(image)
        return torch.argmax(model(as_tensor.unsqueeze(0).to(device))).item()
