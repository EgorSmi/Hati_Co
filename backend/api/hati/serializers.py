from rest_framework import serializers

from .models import HatiMeta


# class BreedDogsSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = BreedDog
#         fields = '__all__'


class HatiMetaSerializer(serializers.ModelSerializer):
    # breed = BreedDogsSerializer(many=True)

    class Meta:
        model = HatiMeta
        exclude = ['id']