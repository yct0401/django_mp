"""
WSGI config for mpdevice project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

# import sys
# sys.path.append('D:\Desktop\Dummy_Device_for_fingertalk\mediapipedevice\mpdevice')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mpdevice.settings')

application = get_wsgi_application()
