from django.views.generic import TemplateView

from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import status

from .models import HatiMeta, HatiInfo
from .serializers import HatiMetaSerializer
from .service import data_processing, data_filter, get_testdata



class HatiMetaView(GenericAPIView):
    """
    Получить мета информацию связанную с объявлениями для поиска собак
    """

    queryset = HatiMeta.objects.all()
    serializer_class = HatiMetaSerializer

    def get(self, request, *args, **kwargs):
        model = self.queryset.model()
        serializer = self.serializer_class

        model.save()

        model.meta_animals()
        model.meta_tails()
        model.meta_colors()
        model.meta_breeds()

        meta = serializer(model).data

        model.delete()

        return Response(meta)


class LookMapView(TemplateView):
    """
    Просмотр карты с камерами для датасета
    """

    template_name = "map_dataset.html"


class HatiFindView(GenericAPIView):
    """
    Получение информации об пропавшей собаки и её поиск
    """

    def post(self, request, *args, **kwargs):
        print(self.request.data)

        odata = {'camera': self.request.data.get('camera'), 'animal': self.request.data.get('animal'),
                 'tail': self.request.data.get('tail'), 'color': self.request.data.get('color'),
                 'radius': self.request.data.get('radius'), 'breed': self.request.data.get('breed'),
                 'marks': self.request.data.get('marks')}


        odata['camera'] = odata['camera'] if odata['camera'] else 'undefined'

        id = self.request.data.get('id')
        is_testdata = self.request.data.get('is_testdata')

        if is_testdata == "true":
            obj_data = get_testdata()
            pass
        else:
            obj_data = HatiInfo.objects.get(id=id).info

        data = data_filter(obj_data, odata)

        return Response({"images":data}, status=status.HTTP_200_OK)


class UploadPhotoView(GenericAPIView):
    """
    Сохранение изображений собаки и её поиск
    """

    def post(self, request, *args, **kwargs):
        files = []
        for file in self.request.data:
            if str(file).startswith('image_'):
                files.append(self.request.data.get(str(file)))

        data = data_processing(files)

        info = {}

        if data:
            obj = HatiInfo.objects.create(info = {"hi"})
            obj.save()
            info = {'id':obj.id}

        return Response(info, status=status.HTTP_200_OK)
