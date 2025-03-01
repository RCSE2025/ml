import pytesseract
import cv2
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
from torchvision import transforms, models
from PIL import Image
from torch.autograd import Variable
from itertools import islice
import warnings

import clip

torch.backends.quantized.engine = 'qnnpack'


device = 'cpu'
config = '-l rus+eng'

emergency_flag = 0
path = './pytorch_nsfw_model/ResNet50_nsfw_model.pth'
classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
exceptions = ['drawings', 'neutral']
text_list = ['Photo without violence and killing a person or animal',
             'Photo with violence and killing of a person or animal or blood']

transformation = transforms.Compose([transforms.Resize(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
                                     ])

# ResNet 50 model for NSFW detecting
try:
    image_model = models.resnet50()
    image_model.fc = nn.Sequential(nn.Linear(2048, 512),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(512, 10),
                                   nn.LogSoftmax(dim=1))
    image_model.load_state_dict(
        torch.load(path, map_location=device))
    image_model.eval()

except Exception:
    warnings.warn("An error occurred while loading the ResNet model")
    emergency_flag = 1

# CLIP model for violence detecting
try:
    clip_model, clip_transform = clip.load(
        "ViT-B/32", device=device)
except RuntimeError:
    warnings.warn("An error occurred while loading the Clip model")
    emergency_flag = 1

# NLP model for detecting aggression in text
try:
    nlp_tokenizer = AutoTokenizer.from_pretrained(
        'nlp_tokenizer/')
    nlp_model = torch.jit.load("nlp_model/pytorch_model_traced.pt")
except Exception:
    warnings.warn("An error occurred while loading the NLP model")
    emergency_flag = 1


class Model:
    def __init__(self, path_list, text_from_post):
        self.emergency_flag = emergency_flag

        self.text_from_post = text_from_post
        self.path_list = path_list
        self.pil_images = [Image.open(x).convert('RGB') for x in self.path_list]

        self.img = [cv2.imread(x) for x in self.path_list]
        self.img = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in self.img]

        # detecting text in an image using PyTesseract model
        try:
            self.string = ' '.join(
                [pytesseract.image_to_string(x, config=config).replace('\n', ' ').strip() for x in self.img])
            self.texts = [self.string, self.text_from_post]
        except Exception:
            warnings.warn("An error occurred while using PyTesseract model")
            self.emergency_flag = 1

        self.tokenized_text = nlp_tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        self.tokenized_text = dict(
            map(lambda x: (x[0], (x[1]).to(device)), self.tokenized_text.items()))

    def predict_image(self, image):
        image_tensor = transformation(image).float()
        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor.cuda()

        input_ = Variable(image_tensor)
        output = image_model(input_)

        return torch.softmax(output[:, :5], 1)

    def get_logits(self, images_path_list, text):
        clip_images = [Image.open(x).convert('RGB') for x in images_path_list]
        transformed_images = [clip_transform(
            x).unsqueeze(0).to(device) for x in clip_images]
        tokenized_text = clip.tokenize(text).to(device)
        with torch.no_grad():
            output = torch.cat([clip_model(image, tokenized_text)[
                               0] for image in transformed_images], dim=0)
        probs = torch.softmax(output, dim=1)[:, 1].cpu().tolist()
        return probs

    def predict(self):
        self.image_dict = [self.predict_image(
            Image.open(x).convert('RGB')) for x in self.path_list]
        self.image_dict = dict(zip(
            map(os.path.basename, self.path_list), [dict(zip(classes, x[0].float().tolist())) for x in self.image_dict]))
        self.logits = nlp_model(
            **self.tokenized_text
        )['logits']

        if len(self.path_list) > 0:
            violence = self.get_logits(self.path_list, text_list)
            for name, proba in zip(map(os.path.basename, self.path_list), violence):
                self.image_dict[name]['violence'] = proba

        self.image_dict['text'] = {}
        self.image_dict['text']['image_text_agressive'], \
            self.image_dict['text']['main_text_agressive'] = torch.softmax(self.logits, 1)[
            :, 1].tolist()
        return self.image_dict

    def moderate(self):

        treshold = 0.6

        if self.emergency_flag:
            return {
                "result": -1,
                "details": {}
            }
        flag = 0
        final_dict = self.predict()
        for _, dict_ in dict(islice(final_dict.items(), len(final_dict) - 1)).items():
            for type_, proba in dict_.items():
                if type_ not in exceptions:
                    if proba > treshold:
                        flag = 1
        if final_dict['text']['image_text_agressive'] > treshold \
                or final_dict['text']['main_text_agressive'] > treshold:
            flag = 1
        return {
            "result": flag,
            "details": final_dict
        }
