from utils.plots import Annotator, colors
import os
import cv2
import numpy as np
import pickle
from copy import deepcopy
import pprint
from PIL import ImageDraw, Image, ImageFont

if __name__ == '__main__':
    with open('../output.pickle', 'rb') as f:
        data = pickle.load(f)
    # print(data)
    for file in os.listdir('../input'):
        imgdata = {'camera': data['cams'][file], 'dogs': {}}
        for elem in data['dogs']:
            if data['dogs'][elem]['imgname'] == file:
                imgdata['dogs']['dog' + ('0' if elem.split('_')[1][0] == '.' else str(
                    int(elem.split('_')[1].split('.')[0]) - 1))] = deepcopy(data['dogs'][elem])
                del imgdata['dogs']['dog' + ('0' if elem.split('_')[1][0] == '.' else str(
                    int(elem.split('_')[1].split('.')[0]) - 1))]['imgname']
        imgdata = pprint.pformat(imgdata)
        f = open('../input/' + file, "rb")
        chunk = f.read()
        f.close()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        try:
            with open('./runs/detect/exp/labels/' + file[:-4] + '_.txt') as f:
                dgs = f.read().split('\n')[:-1]
        except:
            dgs = []
        try:
            with open('./runs/detect/exp2/labels/' + file[:-4] + '_.txt') as f:
                ids = f.read().split('\n')[:-1]
        except:
            ids = []
        for elem in ids:
            dgs.append(str(int(elem.split(' ')[0]) + 5) + elem[1:])
        height, width, _ = img.shape
        labels = ['dog', 'human', 'human with dog', 'bird', 'cat', 'id', 'datetime', 'address']
        annotator = Annotator(img, line_width=1,
                              example=str(labels))
        cnt = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(dgs)):
            annotator.box_label(
                [int(float(dgs[i].split(' ')[1]) * width) - int(float(dgs[i].split(' ')[3]) * width / 2),
                 int(float(dgs[i].split(' ')[2]) * height) - int(float(dgs[i].split(' ')[4]) * height / 2),
                 int(float(dgs[i].split(' ')[1]) * width) + int(float(dgs[i].split(' ')[3]) * width / 2),
                 int(float(dgs[i].split(' ')[2]) * height) + int(float(dgs[i].split(' ')[4]) * height / 2)],
                labels[int(dgs[i].split(' ')[0])] + str(cnt[int(dgs[i].split(' ')[0])]),
                color=colors(int(dgs[i].split(' ')[0]), True))
            cnt[int(dgs[i].split(' ')[0])] += 1
        img = annotator.result()
        font = ImageFont.truetype("arial.ttf", 12)
        w, h = ImageDraw.Draw(Image.new('RGB', (100, 100))).textsize(imgdata, font)
        img = cv2.copyMakeBorder(img, 0, 0, w + 10, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), imgdata, font=font, align="left")
        cv2.imwrite('../input_plotted/file.jpg', np.asarray(img))
        os.rename('../input_plotted/file.jpg', '../input_plotted/' + file)
