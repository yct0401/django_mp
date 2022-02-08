# Generated by Django 3.2.11 on 2022-02-08 07:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rtsp', '0004_alter_device_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='device',
            name='complexity',
            field=models.DecimalField(decimal_places=2, default=0, max_digits=4),
        ),
        migrations.AddField(
            model_name='device',
            name='confidence',
            field=models.DecimalField(decimal_places=2, default=0.5, max_digits=4),
        ),
        migrations.AddField(
            model_name='device',
            name='model',
            field=models.CharField(default='Hands', max_length=20),
            preserve_default=False,
        ),
    ]
