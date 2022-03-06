# Generated by Django 3.2.11 on 2022-03-06 02:02

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='project',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='RTSP',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=50)),
                ('location', models.CharField(max_length=100)),
                ('url', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='device',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=50)),
                ('model', models.CharField(max_length=20)),
                ('complexity', models.DecimalField(decimal_places=2, default=0, max_digits=4)),
                ('confidence', models.DecimalField(decimal_places=2, default=0.5, max_digits=4)),
                ('uuid', models.CharField(max_length=36, null=True)),
                ('project', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='rtsp.project')),
                ('rtsp', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='rtsp.rtsp')),
            ],
        ),
    ]
