from yolov5.detect import run
from yolov5.detect_id import run_id
from shutil import rmtree
import cv2
import pytesseract
from shutil import copyfile
from yandex_geocoder import Client
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import pickle
import json


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
def classificator(img_folder_path, model_path):
    def make_pred(img_path):
        # Prepare the image
        img = Image.open(img_path).resize(img_size, Image.ANTIALIAS)
        img_array = image.img_to_array(img)
        final_image = np.expand_dims(img_array, axis=0)

        # Make a prediction
        pred = np.argmax(model.predict(final_image))
        prediction_name = classes['classes'].iloc[pred]
        return prediction_name

    classes = pd.read_csv(os.path.join(model_path, 'classes.csv'))
    img_size = (224, 224)
    predicts = dict()
    model = keras.models.load_model(
        model_path, custom_objects=None, compile=True, options=None
    )

    # Ignore hidden files
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
    line = line.replace(' ', '_').replace('__', '_').replace('&0', 'AO').replace('Q', 'O').replace('$', 'S').replace(
        'A0', 'AO').replace('OO', 'O').replace('&I', 'T').replace('5A', 'SA').replace('PN', 'PVN').replace('ae_',
                                                                                                           '').replace(
        'SVN', 'PVN').replace('wa_', '').replace('>VI', 'PV').replace('a_', '').replace('PUN', 'PVN').replace('WN',
                                                                                                              'PVN').replace(
        'wwe_', '').replace('EVN', '').replace('.', '').replace("'", '').replace('|', '').replace(')', '').replace('(',
                                                                                                                   '').replace(
        'uPVN', 'PVN').replace('-', '_').replace('Ind', 'hd').replace('G', 'O').replace('/', '1').replace('\\',
                                                                                                          '').replace(
        '&', '8').replace('PVIN', 'PVN').replace('SWAO', 'SVAO').replace('>V', 'P')
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


def run_win():
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-4].replace('_', '') + '_.jpg')
    # try:
    # Очищаем временные файлы с прошлого запуска
    for file in os.listdir('./temp_data/'):
        try:
            rmtree('./temp_data/' + file)
        except:
            os.remove('./temp_data/' + file)

    # запускаем yolov5 на поиск собак, их хозяев, etc
    run(source='./input/', weights='./weights/yolo-enter.pt', imgsz=1280, save_crop=True, nosave=True,
        dirpad10='./temp_data/yolo_crops10/')

    forvalcams = {}

    # получаем собак с хозяевами
    withowners = []
    for file in os.listdir('./temp_data/yolo_crops10/humanwithdog/'):
        withowners.append(file.split('_')[0] + '_.jpg')

    animalthere = set()
    if os.path.exists('./temp_data/yolo_crops10/bird/'):
        for file in os.listdir('./temp_data/yolo_crops10/bird/'):
            imgname = file.split('_')[0] + '_.jpg'
            animalthere.add(imgname)
    if os.path.exists('./temp_data/yolo_crops10/cat/'):
        for file in os.listdir('./temp_data/yolo_crops10/cat/'):
            imgname = file.split('_')[0] + '_.jpg'
            animalthere.add(imgname)

    # начинаем формировать словарь с параметрами собак
    camswithdogs = set()
    dogs = {}
    for file in os.listdir('./temp_data/yolo_crops10/dog/'):
        imgname = file.split('_')[0] + '.jpg'
        camswithdogs.add(imgname)
        animalthere.add(imgname)
        withowner = imgname in withowners
        dogs[file] = {'imgname': imgname, 'withowner': withowner}

    # запускаем yolov5 на поиск id и адресов камер, дат и времени запечатления снимка
    run_id(source='./input/', weights='./weights/yolo-id.pt', imgsz=1280, save_crop=True, nosave=True,
           dirpad0='./temp_data/id_crops/')

    # инициализируем tesseract-OCR
    custom_config_ru = r'--oem 3 --psm 7 -l eng -l rus'
    custom_config_en = r'--oem 3 --psm 7 -l eng'
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
                # img = cv2.imread('./temp_data/id_crops/' + classb + file)
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

    # запускаем три отдельных классификатора на базе EfficentNet для определения цвета, породы и длины хвоста
    breeds = classificator('./temp_data/yolo_crops10/dog/', './weights/model_breeds/')
    colors = classificator('./temp_data/yolo_crops10/dog/', './weights/model_colors/')
    tails = classificator('./temp_data/yolo_crops10/dog/', './weights/model_tails/')

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
        # cams[cam]['pred_coords'] = geocode(cams[cam]['address']) # Раскомментируйте эту строку для улучшения предсказаний с помощью яндекс геокодинга

    for dog in dogs.keys():
        forvalcams[dogs[dog]['imgname']]['isitadog'] = True
        if dogs[dog]['withowner']:
            forvalcams[dogs[dog]['imgname']]['withowner'] = True

    # забиваем полученные предсказания в словарь с данными о собаках
    for dog in dogs:
        dogs[dog]['breed'] = breeds[dog]
        dogs[dog]['color'] = colors[dog]
        dogs[dog]['tail'] = tails[dog]
        forvalcams[dogs[dog]['imgname']]['color'] = colors[dog]
        forvalcams[dogs[dog]['imgname']]['tail'] = tails[dog]

    # добавляем адреса
    cams = geo_addition(cams)

    for cam in cams.keys():
        if len(cams[cam]['true_address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['true_address'][0]
        elif len(cams[cam]['address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['address'][0]
        else:
            forvalcams[cam]['address'] = 'NaN'
        if len(cams[cam]['true_cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['true_cam_id'][0]
        elif len(cams[cam]['cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['cam_id'][0]
        else:
            forvalcams[cam]['id'] = 'NaN'

    for file in os.listdir('./input/'):
        if file[:-5] + '.jpg' not in cams.keys():
            cams[file[:-5] + '.jpg'] = {'isanimalthere': False, 'isitadog': False, 'address': '', 'id': ''}

    csv = 'filename,is_animal_there,is_it_a_dog,is_the_owner_there,color,tail,address,cam_id\n'
    cols = {'dark': '1', 'bright': '2', 'multicolor': '3'}
    tail = {'short_tail': '1', 'long_tail': '2'}
    for cam in forvalcams.keys():
        csv += cam + ','
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
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-5] + '.jpg')


def run_win_back():
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-4].replace('_', '') + '_.jpg')
    # try:
    # Очищаем временные файлы с прошлого запуска
    for file in os.listdir('./temp_data/'):
        try:
            rmtree('./temp_data/' + file)
        except:
            os.remove('./temp_data/' + file)

    # запускаем yolov5 на поиск собак, их хозяев, etc
    run(source='./input/', weights='./weights/yolo-enter.pt', imgsz=1280, save_crop=True, nosave=True,
        dirpad10='./temp_data/yolo_crops10/')

    forvalcams = {}

    # получаем собак с хозяевами
    withowners = []
    for file in os.listdir('./temp_data/yolo_crops10/humanwithdog/'):
        withowners.append(file.split('_')[0] + '_.jpg')

    animalthere = set()
    if os.path.exists('./temp_data/yolo_crops10/bird/'):
        for file in os.listdir('./temp_data/yolo_crops10/bird/'):
            imgname = file.split('_')[0] + '_.jpg'
            animalthere.add(imgname)
    if os.path.exists('./temp_data/yolo_crops10/cat/'):
        for file in os.listdir('./temp_data/yolo_crops10/cat/'):
            imgname = file.split('_')[0] + '_.jpg'
            animalthere.add(imgname)

    # начинаем формировать словарь с параметрами собак
    camswithdogs = set()
    dogs = {}
    for file in os.listdir('./temp_data/yolo_crops10/dog/'):
        imgname = file.split('_')[0] + '.jpg'
        camswithdogs.add(imgname)
        animalthere.add(imgname)
        withowner = imgname in withowners
        dogs[file] = {'imgname': imgname, 'withowner': withowner}

    # запускаем yolov5 на поиск id и адресов камер, дат и времени запечатления снимка
    run_id(source='./input/', weights='./weights/yolo-id.pt', imgsz=1280, save_crop=True, nosave=True,
           dirpad0='./temp_data/id_crops/')

    # инициализируем tesseract-OCR
    custom_config_ru = r'--oem 3 --psm 7 -l eng -l rus'
    custom_config_en = r'--oem 3 --psm 7 -l eng'
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
                # img = cv2.imread('./temp_data/id_crops/' + classb + file)
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

    # запускаем три отдельных классификатора на базе EfficentNet для определения цвета, породы и длины хвоста
    breeds = classificator('./temp_data/yolo_crops10/dog/', './weights/model_breeds/')
    colors = classificator('./temp_data/yolo_crops10/dog/', './weights/model_colors/')
    tails = classificator('./temp_data/yolo_crops10/dog/', './weights/model_tails/')

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
        # cams[cam]['pred_coords'] = geocode(cams[cam]['address']) # Раскомментируйте эту строку для улучшения предсказаний с помощью яндекс геокодинга

    for dog in dogs.keys():
        forvalcams[dogs[dog]['imgname']]['isitadog'] = True
        if dogs[dog]['withowner']:
            forvalcams[dogs[dog]['imgname']]['withowner'] = True

    # забиваем полученные предсказания в словарь с данными о собаках
    for dog in dogs:
        dogs[dog]['breed'] = breeds[dog]
        dogs[dog]['color'] = colors[dog]
        dogs[dog]['tail'] = tails[dog]
        forvalcams[dogs[dog]['imgname']]['color'] = colors[dog]
        forvalcams[dogs[dog]['imgname']]['tail'] = tails[dog]

    # добавляем адреса
    cams = geo_addition(cams)

    for cam in cams.keys():
        if len(cams[cam]['true_address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['true_address'][0]
        elif len(cams[cam]['address']) > 0:
            forvalcams[cam]['address'] = cams[cam]['address'][0]
        else:
            forvalcams[cam]['address'] = 'NaN'
        if len(cams[cam]['true_cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['true_cam_id'][0]
        elif len(cams[cam]['cam_id']) > 0:
            forvalcams[cam]['id'] = cams[cam]['cam_id'][0]
        else:
            forvalcams[cam]['id'] = 'NaN'

    csv = 'filename,is_animal_there,is_it_a_dog,is_the_owner_there,color,tail,address,cam_id\n'
    cols = {'dark': '1', 'bright': '2', 'multicolor': '3'}
    tail = {'short_tail': '1', 'long_tail': '2'}
    for cam in forvalcams.keys():
        csv += cam + ','
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
    # with open('preds.csv', 'w', encoding='utf-8') as f:
    #     f.write(csv)

    # формируем выходной словарь, сохраняем его
    out = {'cams': cams, 'dogs': dogs}
    # with open('./output.pickle', 'wb') as f:
    #     pickle.dump(out, f)
    for file in os.listdir('./input/'):
        os.rename('./input/' + file, './input/' + file[:-5] + '.jpg')

    back_outp = {}
    print(list(cams.keys()))
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
        back_outp[cam]['isitadog'] = cams[cam]['isitadog']
    return back_outp


if __name__ == '__main__':
    run_win()
