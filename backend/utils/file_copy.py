from shutil import copyfile


def make_copy(file):
    copyfile(file, 'model/input')
