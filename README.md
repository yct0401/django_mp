# django_mp

## In Linux System 
commemt out `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP` in `rtsp/view.py`
``` python
...
p[device_name].process=subprocess.Popen(  [   'python',
                                        dai_path,
                                        CAM[rtsp_name].url, 
                                        device_name, 
                                        model, 
                                        str(complexity), 
                                        str(confidence),
                                        device_uuid ],
                                        stdin=p[device_name].pipe_r,
                                        shell=True, 
                                        # creationflags=subprocess.CREATE_NEW_PROCESS_GROUP <- this flag only for Windows
                                       )
...
```
## start commad

`python manage.py runserver --settings=mpdevice.settings`

## run dai

`python mpdai.py <rtsp_url> <device_name> <model> <complexity [0,1] > <confidence [0-100] > <uuid>`

