import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1620, 512)
        self.fc2 = nn.Linear(512, 12)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

device = torch.device('cpu')
model = Net().to(device)
model.load_state_dict(torch.load("photomath5.pth",map_location=device))
model.eval()

def getExpression(slike):
    image = numpy.array(slike)
    image = image[:, :, ::-1].copy()
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    coord = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        if h > 0.6 * image.shape[0] or w > 0.6 * image.shape[1]:
            continue
        if h < 0.05 * image.shape[0] and w < 0.05 * image.shape[1]:
            continue
        coord.append((x, y, w, h))

    coord.sort(key=lambda tup: tup[0])

    count = 0
    slike = []
    for cor in coord:
        [x, y, w, h] = cor
        t = image[y:y + h, x:x + w, :]
        slike.append(Image.fromarray(t))
        count += 1
    print("number of char in image:", count)
    return slike

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize((45,45)),transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(), transforms.Normalize((0.456,), (0.224,))])
    for im in range(0,len(image_bytes)):
        image_bytes[im] = transform(image_bytes[im]).unsqueeze_(0)
    return image_bytes

def get_prediction(slike):
    listaZnamenki = []
    stringer = ""
    listaKrajnjih = []
    broj = ""
    listaOp = []
    for slika in slike:
        slika = slika.to(device)
        outputs = model(slika)
        print(outputs)
        rez = int(torch.argmax(outputs.data))
        if rez == 10:
            listaZnamenki.append('+')
            stringer += '+'
            listaKrajnjih.append(int(broj))
            broj = ""
            listaOp.append("+")
        elif rez == 11:
            listaZnamenki.append('-')
            stringer += '-'
            listaKrajnjih.append(int(broj))
            broj = ""
            listaOp.append("-")
        else:
            listaZnamenki.append(str(rez))
            stringer += str(rez)
            broj += str(rez)
    if broj != "":
        listaKrajnjih.append(int(broj))
    brojac = 0
    brojacOp = 0
    rez = listaKrajnjih[brojac]
    while brojac < len(listaKrajnjih)-1:
        br2 = listaKrajnjih[brojac+1]
        if listaOp[brojacOp] == "+":
            rez = rez+br2
        else:
            rez = rez-br2
        brojacOp+= 1
        brojac += 1
    return rez, stringer


