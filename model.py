from yolov5.detect import run
from yolov5.detect_id import run_id
from shutil import rmtree
import cv2
import pytesseract
from yandex_geocoder import Client
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import json
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score
import pytorch_lightning as pl


# получаем адреса камер из id
def geo_addition(data):
    path_2_dvn = './cam_data/data-8180-2021-08-18.json'
    path_2_pvn = './cam_data/data-49169-2021-08-12.json'
    path_2_mmc = './cam_data/data-8174-2021-08-10.json'  # можно не использовать, но там немного камер
    with open(path_2_dvn, 'rt', encoding='cp1251') as file:
        dvn = json.load(file)

    with open(path_2_pvn, 'rt', encoding='cp1251') as file:
        pvn = json.load(file)

    with open(path_2_mmc, 'rt', encoding='cp1251') as file:
        mmc = json.load(file)

    reestr = pvn + dvn + mmc

    for img in data.keys():
        cam_ids = data[img]["cam_id"]
        true_ids = []
        true_adresses = []
        true_coords = []
        for cam_id in cam_ids:
            true_id = False
            for real in reestr:
                if cam_id == real["ID"]:
                    true_id = True
                    true_adress = real["Address"]
                    true_coord = real["geoData"]["coordinates"]
                    break
            if true_id:
                true_ids.append(cam_id)
                true_adresses.append(true_adress)
                true_coords.append(true_coord)

        data[img]["true_cam_id"] = true_ids
        data[img]["true_address"] = true_adresses
        data[img]["true_coords"] = true_coords

    return data


# универсальный классификатор
def classificator(img_folder_path, model_path, model):
    def make_pred(img_path):
        # подготовка изображения
        img = Image.open(img_path).resize(img_size, Image.ANTIALIAS)
        img_array = image.img_to_array(img)
        final_image = np.expand_dims(img_array, axis=0)

        # предсказываем
        pred = np.argmax(model.predict(final_image))
        prediction_name = classes['classes'].iloc[pred]
        return prediction_name

    classes = pd.read_csv(os.path.join(model_path, 'classes.csv'))
    img_size = (224, 224)
    predicts = dict()
    # model = keras.models.load_model(
    #     model_path, custom_objects=None, compile=True, options=None
    # )

    # игнорируем скрытые файлы
    images = [x for x in os.listdir(img_folder_path) if not x.startswith(".")]

    for img in images:
        prediction = make_pred(os.path.join(img_folder_path, img))
        predicts[img] = prediction

    return predicts


# постобработка id камер после тессеракта
def tess_post(instr):
    bads = ['Apxus ', 'Apo ', 'Apoate ', 'Apoais ', '', 'Apxne ', 'Apis ', 'xe ', '| ', 'Apres ', 'Apne ', 'Apoas ',
            'Apos ', 'Apxuwe ', 'Apoaie ', 'Apoare ', 'Apxiwe ', 'Apne ', 'Apxug ', '‘', 'Apoats ', '‘Apatite ',
            'Apatite ', 'Apoae ', 'Ba i See oe', '<ue ', 'AP', 'So i LA Se ed', 'SARE OE Ne', '\'', 'Bi i Sas aa',
            ':PRE eee ees', 'Bi i iS eee', 'ATE Ce Eee', 'PRE eee ees', 'Ss cia i i', 'PEL a', 'PRE eee ees', 'xwe ',
            'id ', 'SE Se ec ea', 'aa ', '13,', 'cen cso eee rena tiie ence see mien AEE', 'ue ', '_pxuie ', 'pxue ',
            '4426_6', 'TE INES SRE SEI', 'SETS ae', 'al', 'SRE TS He aes', 'SETS er a ere', '12 cent_ 2021, 03:32:54',
            'SRT ES er ae', 'Pnowanka nepen nonbezqom Nel', 'Bo i i Scat aL', 'Bh lS i td', '1 ', '/&0_1461_2', '°',
            ',']
    line = instr
    for i in bads:
        line = line.replace(i, '')
    line = line.replace('  ', ' ').replace('¥', 'V')
    if line[-1:] == '_' or line[-1:] == ' ':
        line = line[:-1]
    line = line.replace(' ', '_').replace(':', '_').replace('__', '_').replace('&0', 'AO').replace('Q', 'O').replace(
        '$', 'S').replace(
        'A0', 'AO').replace('OO', 'O').replace('&I', 'T').replace('5A', 'SA').replace('PN', 'PVN').replace('ae_',
                                                                                                           '').replace(
        'SVN', 'PVN').replace('wa_', '').replace('>VI', 'PV').replace('a_', '').replace('PUN', 'PVN').replace('WN',
                                                                                                              'PVN').replace(
        'wwe_', '').replace('EVN', '').replace('.', '').replace("'", '').replace('|', '').replace(')', '').replace('(',
                                                                                                                   '').replace(
        'uPVN', 'PVN').replace('-', '_').replace('Ind', 'hd').replace('G', 'O').replace('/', '1').replace('\\',
                                                                                                          '').replace(
        '&', '8').replace('PVIN', 'PVN').replace('SWAO', 'SVAO').replace('>V', 'P').replace('1PVN', 'PVN').replace(
        'B_PVN', 'PVN').replace('13_PVN', 'PVN').replace('”', '').replace('SUAD', 'SVAO').replace('mane_PYN',
                                                                                                  'PVN').replace(
        'manPYN', 'PVN').replace('PYN', 'PVN').replace('=_PVN', 'PVN').replace('ce_PVN', 'PVN').replace('8_PVN',
                                                                                                        'PVN').replace(
        '1e_PVN', 'PVN').replace('VIN', 'PVN').replace('panPVN', 'PVN').replace('ane_PVN', 'PVN').replace('PYN',
                                                                                                          'PVN').replace(
        'PvN', 'PVN').replace('URO', 'UAO').replace('smPVN', 'PVN').replace('me_PVN', 'PVN').replace('WAO',
                                                                                                     'VAO').replace(
        '1PARK', 'PARK').replace('1PVN', 'PVN').replace('1g_PVN', 'PVN').replace('4PVN', 'PVN').replace('mPVN',
                                                                                                        'PVN').replace(
        'fe_PVN', 'PVN').replace('we_PVN', 'PVN').replace('8_UVN', 'UVN').replace('1s_PVN', 'PVN').replace('@_PARK',
                                                                                                           'PARK')
    if line[:2] == 'VN':
        line = 'P' + line
    if line[-1:] == '_':
        line = line[:-1]
    return line


# геокодинг через яндекс api
def geocode(arr):
    client = Client("your-id")
    ans = []
    for elem in arr:
        try:
            coords = client.coordinates(elem)
            ans.append([float(coords[0]), float(coords[1])])
        except:
            pass
    return ans


# класс для классификатора-трансформера
class TestDatasetViT(Dataset):
    def __init__(self, data_path, imgs, feature_extractor=None):
        self.data_path = data_path
        self.imgs = imgs
        self.W = 224
        self.H = 224
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.imgs)

    # load image from disk
    def _load_train_image(self, fn):
        # мб добавить cv.resize()
        f = open(os.path.join(self.data_path, fn), "rb")
        chunk = f.read()
        f.close()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        # img = cv.imread(filename=os.path.join(self.data_path, fn))
        img = cv.resize(img, (self.W, self.H), interpolation=cv.INTER_AREA)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.float32(img) / 255

        return img

    def _load_train_image_for_extractor(self, fn):
        f = open(os.path.join(self.data_path, fn), "rb")
        chunk = f.read()
        f.close()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        # img = cv.imread(filename=os.path.join(self.data_path, fn))
        img = cv.resize(img, (self.W, self.H), interpolation=cv.INTER_AREA)
        return img

    def __getitem__(self, idx):
        image_name = self.imgs[idx]
        image = self._load_train_image_for_extractor(self.imgs[idx])
        inpt = self.feature_extractor(images=image, return_tensors="pt")
        channels = 3
        pixel_values = inpt['pixel_values'].view(3, self.feature_extractor.size, self.feature_extractor.size)
        image = self._load_train_image(self.imgs[idx])
        sample = image_name, torch.tensor(image).type(torch.float), pixel_values
        return sample


# класс для классификатора-трансформера
class CollarTagger(pl.LightningModule):
    def __init__(self, n_classes: int, ViT_model, n_training_steps=None, n_warmup_steps=None, n_hidden_layers_cat=12):
        super().__init__()
        self.n_hidden_layers_cat = n_hidden_layers_cat
        self.ViT_model = ViT_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.ViT_model.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, imgs, pixel_values, labels=None):
        output = self.ViT_model(pixel_values)
        last_hidden_state = output.last_hidden_state
        pooler_output = output.pooler_output  # but we concat last hidden_layers
        # print('Pooler output: ', pooler_output.shape)
        hidden_states = output.hidden_states
        # print('Hidden_states ', len(hidden_states))
        summed_last_cat_layers = torch.stack(hidden_states[-self.n_hidden_layers_cat:]).sum(0)
        pooled_vector = torch.mean(summed_last_cat_layers, dim=1)  # may be better than sum(), or we can use max-pooling
        # print('pooled_vector: ', pooled_vector.shape)
        pooled_vector = self.dropout(pooled_vector)
        output = self.classifier(pooled_vector)
        # сразу сделаем reshape
        output = output.view(-1)
        # print('out: ', output.shape)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}

    def validation_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        imgs = batch[0]
        features = batch[1]
        labels = batch[2]
        loss, outputs = self.forward(imgs, features, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        preds = [1 if x > 0.0 else 0 for x in outputs]
        return {"loss": loss, "predictions": preds, "labels": labels}

    def training_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        roc_auc = roc_auc_score(predictions, labels)
        self.log(f"roc_auc/Train", roc_auc, self.current_epoch)
        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Train", accuracy, self.current_epoch)

    def validation_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Val", accuracy, self.current_epoch)

    def test_epoch_end(self, outputs):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"]:
                predictions.append(out_predictions)

        labels = torch.tensor(labels, dtype=int)
        predictions = torch.tensor(predictions, dtype=int)

        accuracy = accuracy_score(predictions, labels)
        self.log(f"Acc/Test", accuracy, self.current_epoch)

    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


# загрузка классификатора-трансформера в оперативную память
def load_model(Path_2_model="../input/model-vit/model_vit_collars.pt"):
    # const
    n_classes = 1
    MODEL_NAME = 'google/vit-base-patch16-224-in21k'
    vit_model = ViTModel.from_pretrained(MODEL_NAME, return_dict=True, output_hidden_states=True)
    warmup_steps, total_training_steps = 26, 133

    # model
    model = CollarTagger(n_classes=n_classes, ViT_model=vit_model, n_warmup_steps=warmup_steps,
                         n_training_steps=total_training_steps)

    # load weights
    model.load_state_dict(torch.load(Path_2_model))
    model.eval()
    return model


# предсказать классификатором-трансформером
def collar_classification(test_path, model):
    def get_dataloader(test_path):
        feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        test_imgs = os.listdir(test_path)
        test_dataset = TestDatasetViT(test_path, test_imgs, feature_extractor)
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False)
        return test_loader

    # data
    test_loader = get_dataloader(test_path)

    model.eval()

    # prediction
    prediction = []
    names = []
    for img_name, imgs, features in test_loader:
        names.extend(img_name)
        _, outputs = model(imgs, features, None)
        preds = ['false' if x > 0.0 else 'true' for x in outputs]
        prediction.extend(preds)

    return dict(zip(names, prediction))


# запустить модель
def run_model(linux=True, back=True, yandex=False):
    # добавляем нижнее подчёркивание после имени каждого файла для корректной работы yolo
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-4].replace('_', '') + '_.jpg')
    # Очищаем временные файлы с прошлого запуска
    for file in os.listdir('./temp_data/'):
        try:
            rmtree('./temp_data/' + file)
        except:
            os.remove('./temp_data/' + file)

    # запускаем yolov5 на поиск собак, их хозяев, etc
    run(source='./input/', weights='./weights/yolo-enter.pt', imgsz=1280, save_crop=True, nosave=True,
        dirpad10='./temp_data/yolo_crops10/')
    # run(source='./input/', weights='./weights/yolo-enter.pt', imgsz=1280, save_crop=True, nosave=True,
    #     dirpad10='./temp_data/yolo_crops10/', save_txt=True, save_conf=True)

    forvalcams = {}

    # получаем собак с хозяевами
    withowners = []
    if os.path.exists('./temp_data/yolo_crops10/humanwithdog/'):
        for file in os.listdir('./temp_data/yolo_crops10/humanwithdog/'):
            withowners.append(file.split('_')[0] + '_.jpg')

    # получаем камеры, где есть животные
    animalthere = set()
    if os.path.exists('./temp_data/yolo_crops10/bird/'):
        for file in os.listdir('./temp_data/yolo_crops10/bird/'):
            imgname = file.split('_')[0] + '.jpg'
            animalthere.add(imgname)
    if os.path.exists('./temp_data/yolo_crops10/cat/'):
        for file in os.listdir('./temp_data/yolo_crops10/cat/'):
            imgname = file.split('_')[0] + '.jpg'
            animalthere.add(imgname)

    # начинаем формировать словарь с параметрами собак
    camswithdogs = set()
    dogs = {}
    if os.path.exists('./temp_data/yolo_crops10/dog/'):
        for file in os.listdir('./temp_data/yolo_crops10/dog/'):
            imgname = file.split('_')[0] + '.jpg'
            camswithdogs.add(imgname)
            animalthere.add(imgname)
            withowner = imgname[:-4] + '_.jpg' in withowners
            dogs[file] = {'imgname': imgname, 'withowner': withowner}

    # запускаем yolov5 на поиск id и адресов камер, дат и времени запечатления снимка
    run_id(source='./input/', weights='./weights/yolo-id.pt', imgsz=1280, save_crop=True, nosave=True,
           dirpad0='./temp_data/id_crops/')
    # run_id(source='./input/', weights='./weights/yolo-id.pt', imgsz=1280, save_crop=True, nosave=True,
    #        dirpad0='./temp_data/id_crops/', save_txt=True, save_conf=True)

    # инициализируем tesseract-OCR
    custom_config_ru = r'--oem 3 --psm 7 -l eng -l rus'
    custom_config_en = r'--oem 3 --psm 7 -l eng'
    if not linux:
        pytesseract.pytesseract.tesseract_cmd = './tesseract-ocr-win/tesseract.exe'

    # прогоняем тессеракт на вырезках, полученных от yolov5, и сохраняем результаты в словарь
    cams = {}
    for classb in ['address/', 'cam_id/', 'date_time/']:
        if os.path.exists('./temp_data/id_crops/' + classb):
            for file in os.listdir('./temp_data/id_crops/' + classb):
                imgname = file.split('_')[0] + '_.jpg'
                if imgname not in cams.keys():
                    cams[imgname] = {'address': [], 'cam_id': [], 'date_time': []}
                f = open('./temp_data/id_crops/' + classb + file, "rb")
                chunk = f.read()
                f.close()
                chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
                img = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
                thresh = 200
                # зачерняем лишние пиксели для улучшения предиктов тессеракта(т.к. текст на камерах белый)
                img[(img[:, :, 0] < thresh) & (img[:, :, 1] < thresh) & (img[:, :, 2] < thresh)] = [0, 0, 0]
                if classb == 'cam_id/':
                    # здесь используется функция постобработки предиктов тессеракта, объявленная выше
                    cams[imgname][classb[:-1]].append(
                        tess_post(pytesseract.image_to_string(img, config=custom_config_en).split('\n')[0]))
                else:
                    cams[imgname][classb[:-1]].append(
                        pytesseract.image_to_string(img, config=custom_config_ru).split('\n')[0])

    # запускаем три отдельных классификатора на базе EfficentNet и один ViT для определения цвета, породы, длины хвоста и ошейника/намордника/одежды
    breeds = {}
    colors = {}
    tails = {}
    attrs = {}
    if os.path.exists('./temp_data/yolo_crops10/dog/'):
        breeds = classificator('./temp_data/yolo_crops10/dog/', './weights/model_breeds/', breeds_model)
        colors = classificator('./temp_data/yolo_crops10/dog/', './weights/model_colors/', colors_model)
        tails = classificator('./temp_data/yolo_crops10/dog/', './weights/model_tails/', tails_model)
        attrs = collar_classification('./temp_data/yolo_crops10/dog/', attr_model)

    # убираем нижние подчёркивания из названий камер
    camkeys = list(cams.keys())
    for i in camkeys:
        cams[i[:-5] + '.jpg'] = cams[i]
        del cams[i]

    # если OCR неудачно считал id камеры, но успешно или почти успешно справился с расшифровкой адреса, мы можем
    # прибегнуть к геокодингу, чтобы всё равно найти координаты(но мы не уверены, разрешено ли правилами хакатона
    # использование таких условно бесплатных сервисов)
    for cam in cams.keys():
        forvalcams[cam] = {'isanimalthere': cam in animalthere, 'isitadog': False, 'withowner': False}
        cams[cam]['isanimalthere'] = cam in animalthere
        cams[cam]['isitadog'] = False
        cams[cam]['pred_coords'] = []
        if yandex:
            cams[cam]['pred_coords'] = geocode(
                cams[cam]['address'])  # улучшение предсказаний с помощью яндекс геокодинга

    # определяем есть ли собака на фото и с хозяином ли она
    for dog in dogs.keys():
        forvalcams[dogs[dog]['imgname']]['isitadog'] = True
        cams[dogs[dog]['imgname']]['isitadog'] = True
        if dogs[dog]['withowner']:
            forvalcams[dogs[dog]['imgname']]['withowner'] = True

    # забиваем полученные предсказания в словарь с данными о собаках
    for dog in dogs:
        dogs[dog]['breed'] = breeds[dog]
        dogs[dog]['color'] = colors[dog]
        dogs[dog]['tail'] = tails[dog]
        forvalcams[dogs[dog]['imgname']]['color'] = colors[dog]
        forvalcams[dogs[dog]['imgname']]['tail'] = tails[dog]
        forvalcams[dogs[dog]['imgname']]['breed'] = breeds[dog]
        forvalcams[dogs[dog]['imgname']]['attrs'] = attrs[dog]

    # добавляем адреса
    cams = geo_addition(cams)

    # предиктов много, выбираем из них наиболее правдоподобные в заданном порядке
    for cam in cams.keys():
        if len(cams[cam]['true_address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['true_address'][0]
        elif len(cams[cam]['address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['address'][0]
        else:
            forvalcams[cam]['address'] = ''
        if len(cams[cam]['true_cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['true_cam_id'][0]
        elif len(cams[cam]['cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['cam_id'][0]
        else:
            forvalcams[cam]['id'] = ''

    if not back:
        # формируем csv для валидации
        csv = ',filename,is_animal_there,is_it_a_dog,is_the_owner_there,color,tail,address,cam_id\n'
        cols = {'dark': '1', 'bright': '2', 'multicolor': '3'}
        tail = {'short_tail': '1', 'long_tail': '2'}
        itr = 0
        for cam in forvalcams.keys():
            csv += str(itr) + ',' + cam + ','
            itr += 1
            csv += ('1' if forvalcams[cam]['isanimalthere'] else '0') + ','
            csv += ('1' if forvalcams[cam]['isitadog'] else '0') + ','
            if not forvalcams[cam]['isitadog']:
                csv += '0,0,0,'
            else:
                csv += ('1' if forvalcams[cam]['withowner'] else '0') + ','
                csv += cols[forvalcams[cam]['color']] + ','
                csv += tail[forvalcams[cam]['tail']] + ','
            csv += forvalcams[cam]['address'].replace(',', '') + ','
            csv += forvalcams[cam]['id'] + '\n'
        with open('preds.csv', 'w', encoding='utf-8') as f:
            f.write(csv)
        # формируем выходной словарь, сохраняем его
        out = {'cams': cams, 'dogs': dogs}
        with open('./output.pickle', 'wb') as f:
            pickle.dump(out, f)

    # убираем нижние подчёркивания из названий файлов
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-5] + '.jpg')

    if back:
        # специальный формат вывода для бэкенда
        back_outp = {}
        for cam in cams:
            back_outp[cam] = {}
            if len(cams[cam]['true_cam_id']) > 0:
                back_outp[cam]['cam_id'] = cams[cam]['true_cam_id'][0]
            elif len(cams[cam]['cam_id']) > 0:
                back_outp[cam]['cam_id'] = cams[cam]['cam_id'][0]
            else:
                back_outp[cam]['cam_id'] = ''
            if len(cams[cam]['true_coords']) > 0:
                back_outp[cam]['coords'] = cams[cam]['true_coords'][0]
            elif len(cams[cam]['pred_coords']) > 0:
                back_outp[cam]['coords'] = cams[cam]['cam_id'][0]
            else:
                back_outp[cam]['coords'] = ''
            if len(cams[cam]['true_address']) > 0:
                back_outp[cam]['address'] = cams[cam]['true_address'][0]
            elif len(cams[cam]['address']) > 0:
                back_outp[cam]['address'] = cams[cam]['address'][0]
            else:
                back_outp[cam]['address'] = ''
            back_outp[cam]['date_time'] = cams[cam]['date_time']
            back_outp[cam]['isanimalthere'] = cams[cam]['isanimalthere']
            back_outp[cam]['isitadog'] = forvalcams[cam]['isitadog']
            back_outp[cam]['withowner'] = forvalcams[cam]['withowner']
            if forvalcams[cam]['isitadog']:
                back_outp[cam]['breed'] = forvalcams[cam]['breed']
                back_outp[cam]['color'] = forvalcams[cam]['color']
                back_outp[cam]['tail'] = forvalcams[cam]['tail']
                back_outp[cam]['markers'] = forvalcams[cam]['attrs']
            else:
                back_outp[cam]['breed'] = ''
                back_outp[cam]['color'] = ''
                back_outp[cam]['tail'] = ''
                back_outp[cam]['markers'] = ''
        return back_outp


breeds_model = model = keras.models.load_model(
    './weights/model_breeds/', custom_objects=None, compile=True, options=None
)
colors_model = model = keras.models.load_model(
    './weights/model_colors/', custom_objects=None, compile=True, options=None
)
tails_model = model = keras.models.load_model(
    './weights/model_tails/', custom_objects=None, compile=True, options=None
)
attr_model = load_model('weights/model_vit_collars.pt')

if __name__ == '__main__':
    run_model(linux=False, back=False, yandex=False)
