from django.db import models

from .choises import ANIMAL_CHOICES, TAIL_CHOICES, COLOR_CHOICES, BREAD_CHOICES


# class BreedDog(models.Model):
#     name = models.CharField("Название", max_length=100)

#     def __str__(self):
#         return self.name

#     class Meta:
#         verbose_name = "Порода собаки"
#         verbose_name_plural = "Породы собак"


class HatiInfo(models.Model):
    info = models.JSONField(verbose_name="Данные")


class HatiMeta(models.Model):
    animal = models.JSONField("Животное", default=list)
    tail = models.JSONField("Хвост", default=list)
    color = models.JSONField("Цвет", default=list)
    breed = models.JSONField("Порода", default=list)
    # breed = models.ManyToManyField(BreedDog, verbose_name='Породы')

    def meta_animals(self):
        self.animal = [{'id': animal[0], 'name': animal[1]} for animal in ANIMAL_CHOICES]
        return self

    def meta_tails(self):
        self.tail = [{'id': tail[0], 'name': tail[1]} for tail in TAIL_CHOICES]
        return self

    def meta_colors(self):
        self.color = [{'id': color[0], 'name': color[1]} for color in COLOR_CHOICES]
        return self

    def meta_breeds(self):
        self.breed = [{'id': breed[0], 'name': breed[1]} for breed in BREAD_CHOICES]
        # for _breed in BreedDog.objects.all():
        #     self.breed.add(_breed)
        return self