# Generated by Django 3.2.8 on 2021-11-04 06:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('hati', '0005_auto_20211026_0045'),
    ]

    operations = [
        migrations.CreateModel(
            name='HatiInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('info', models.JSONField(verbose_name='Данные')),
            ],
        ),
    ]
