# Generated by Django 3.2.11 on 2022-01-17 20:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rtsp', '0003_auto_20220118_0409'),
    ]

    operations = [
        migrations.AlterField(
            model_name='device',
            name='id',
            field=models.IntegerField(primary_key=True, serialize=False),
        ),
    ]