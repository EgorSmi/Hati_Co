import os
import random
import string
from shutil import copyfile
from model import run_model
import json
import geopy.distance
import pickle

from django_app import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


def geo_addition1(data):
    path_2_dvn = 'cam_data/data-8180-2021-08-18.json'
    path_2_pvn = 'cam_data/data-49169-2021-08-12.json'
    path_2_mmc = 'cam_data/data-8174-2021-08-10.json'
    with open(path_2_dvn, 'rt', encoding='cp1251') as file:
        dvn = json.load(file)

    with open(path_2_pvn, 'rt', encoding='cp1251') as file:
        pvn = json.load(file)

    with open(path_2_mmc, 'rt', encoding='cp1251') as file:
        mmc = json.load(file)

    reestr = pvn + dvn + mmc

    cam_id = data
    true_coord = []
    for real in reestr:
        if cam_id == real["ID"]:
            true_coord = real["geoData"]["coordinates"]
            break

    return true_coord


def data_processing(files):
    # создание изображений
    # paths = []
    for file in files:
        rand_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(14))
        new_file_name = rand_name + '.' + str(file).split(".")[1]
        photo_url = str(settings.BASE_DIR) + '/media/hati/' + new_file_name
        input_url = str(settings.BASE_DIR) + '/input/' + new_file_name
        # media_url = str(settings.BASE_DIR) + '/media/dogs/' + new_file_name
        path = default_storage.save(photo_url, ContentFile(file.read()))
        copyfile(path, input_url)
        # copyfile(path, media_url)
        # paths.append(media_url)
        # os.remove(path)

    # запуск модели
    preds = run_model(linux=True, back=True, yandex=False)

    # удаление изображений
    dir = str(settings.BASE_DIR) + '/input/'
    filelist = [f for f in os.listdir(dir)]
    for f in filelist:
        os.remove(os.path.join(dir, f))

    return preds


def data_filter(preds, odata):
    # обработка списка изображений
    data = []
    for pr in enumerate(preds):
        if odata['camera'] != 'undefined' and preds[pr[1]]['coords'] != '':
            coords_1 = (preds[pr[1]]['coords'][0], preds[pr[1]]['coords'][1])
            crds = geo_addition1(odata['camera'])
            coords_2 = (crds[0], crds[1])
            if geopy.distance.vincenty(coords_1, coords_2).m > odata['radius']:
                continue
        if odata['tail'] != 'undefined' and preds[pr[1]]['tail'] != odata['tail']:
            continue
        if odata['animal'] != 'undefined' and ((preds[pr[1]]['isanimalthere'] and odata['animal'] == 'false') or (
                not preds[pr[1]]['isanimalthere'] and odata['animal'] == 'true')):
            continue
        if odata['isitadog'] != 'undefined' and ((preds[pr[1]]['isitadog'] and odata['isitadog'] == 'false') or (
                not preds[pr[1]]['isitadog'] and odata['isitadog'] == 'true')):
            continue
        if odata['withowner'] != 'undefined' and ((preds[pr[1]]['withowner'] and odata['withowner'] == 'false') or (
                not preds[pr[1]]['withowner'] and odata['withowner'] == 'true')):
            continue
        if odata['breed'] != 'undefined' and preds[pr[1]]['breed'] != odata['breed']:
            continue
        if odata['color'] != 'undefined' and preds[pr[1]]['color'] != odata['color']:
            continue
        if odata['markers'] != 'undefined' and preds[pr[1]]['markers'] != odata['markers']:
            continue

        file_tst = str(settings.BASE_DIR) + '/media/dogs/' + pr[1]
        if os.path.exists(file_tst):
            data.append({'id': pr[0], 'image': 'http://84.201.148.17:26555/media/dogs/' + pr[1],
                         'address': preds[pr[1]]['address']})
        else:
            file_user = str(settings.BASE_DIR) + '/media/hati/' + pr[1]
            if os.path.exists(file_user):
                data.append({'id': pr[0], 'image': 'http://84.201.148.17:26555/media/hati/' + pr[1],
                             'address': preds[pr[1]]['address']})

    return data


def get_testdata():
    with open('output_model.pickle', 'rb') as f:
        data = pickle.load(f)
    return data
