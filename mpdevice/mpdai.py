
import atexit
import logging
import os.path
import signal
import re
import sys
import time
import traceback
from typing import Counter
import mpsa

from threading import Thread, Event, active_count
from uuid import UUID

from iottalkpy import dan
from iottalkpy.color import DAIColor
from iottalkpy.dan import DeviceFeature, NoData
from iottalkpy.exceptions import RegistrationError



log = logging.getLogger(DAIColor.wrap(DAIColor.logger, 'DAI'))
log.setLevel(level=logging.INFO)

class DAI():
    def __init__(self, api_url, device_model, device_addr=None,
                 device_name=None, persistent_binding=False, username=None,
                 extra_setup_webpage='', device_webpage='',
                 register_callback=None, on_register=None, on_deregister=None,
                 on_connect=None, on_disconnect=None,
                 push_interval=1, device_features=None):
        
        self.api_url = api_url
        self.device_model = device_model
        self.device_addr = device_addr
        self.device_name = device_name
        self.persistent_binding = persistent_binding
        self.username = username
        self.extra_setup_webpage = extra_setup_webpage
        self.device_webpage = device_webpage

        self.register_callback = register_callback
        self.on_register = on_register
        self.on_deregister = on_deregister
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect

        self.push_interval = push_interval

        self.device_features = device_features if device_features else {}
        self.flags = {}

    def on_signal(self, signal, df_list):
        log.info('Receive signal: \033[1;33m%s\033[0m, %s', signal, df_list)
        if 'CONNECT' == signal:
            for df_name in df_list:
                # race condition
                if not self.flags.get(df_name):
                    self.flags[df_name] = True
        elif 'DISCONNECT' == signal:
            for df_name in df_list:
                self.flags[df_name] = False
        elif 'SUSPEND' == signal:
            # Not use
            pass
        elif 'RESUME' == signal:
            # Not use
            pass
        return True

    def on_data(self, df_name, data):
        try:
            self.device_features[df_name].on_data(data)
        except Exception:
            traceback.print_exc()
            return False
        return True

    @staticmethod
    def df_func_name(df_name):
        return re.sub(r'-', r'_', df_name)

    def parse_df_profile(self, sa, typ):
        def f(p):
            if isinstance(p, str):
                df_name, param_type = p, None
            elif isinstance(p, tuple) and len(p) == 2:
                df_name, param_type = p
            else:
                raise RegistrationError(
                    'Invalid `{}_list`, usage: [df_name, ...] '
                    'or [(df_name, type), ...]'.format(typ))

            on_data = push_data = getattr(sa, DAI.df_func_name(df_name), None)

            df = DeviceFeature(
                df_name=df_name, df_type=typ, push_data=push_data, on_data=on_data)
            return df_name, df

        profiles = getattr(sa, '{}_list'.format(typ), [])
        return dict(map(f, profiles))

    def _check_parameter(self):
        if self.api_url is None:
            raise RegistrationError('api_url is required')

        if self.device_model is None:
            raise RegistrationError('device_model not given.')

        if isinstance(self.device_addr, UUID):
            self.device_addr = str(self.device_addr)
        elif self.device_addr:
            try:
                UUID(self.device_addr)
            except ValueError:
                try:
                    self.device_addr = str(UUID(int=int(self.device_addr, 16)))
                except ValueError:
                    log.warning('Invalid device_addr. Change device_addr to None.')
                    self.device_addr = None

        if self.persistent_binding and self.device_addr is None:
            msg = ('In case of `persistent_binding` set to `True`, '
                   'the `device_addr` should be set and fixed.')
            raise ValueError(msg)

        if not self.device_features.keys():
            raise RegistrationError('Neither idf_list nor odf_list is empty.')
        
        return True


# main program
print(sys.argv)
t = Thread(target = mpsa.Streaming, args=(sys.argv[1], sys.argv[2]))
t.setDaemon(True)
t.start()

dai = DAI(
    api_url=mpsa.api_url,
    device_model=mpsa.device_model, 
    device_addr=getattr(mpsa, 'device_addr', None),
    device_name=sys.argv[2],
    persistent_binding=False, 
    username=getattr(mpsa, 'username', None),
    extra_setup_webpage='', 
    device_webpage='',
    register_callback=getattr(mpsa, 'register_callback', None), 
    on_register=getattr(mpsa, 'on_register', None), 
    on_deregister=getattr(mpsa, 'on_deregister', None),
    on_connect=getattr(mpsa, 'on_connect', None), 
    on_disconnect=getattr(mpsa, 'on_disconnect', None),
    push_interval=getattr(mpsa, 'push_interval', 1), 
    device_features=None   
)

dai.device_features = dict(
        dai.parse_df_profile(mpsa, 'idf'),
        **dai.parse_df_profile(mpsa, 'odf'))

dai._check_parameter()

idf_list = []
odf_list = []
for df in dai.device_features.values():
    if df.df_type == 'idf':
        idf_list.append(df.profile())
    else:
        odf_list.append(df.profile())

client = dan.Client()

# client.register(
#     url=dai.api_url,
#     on_signal=dai.on_signal,
#     on_data=dai.on_data,
#     accept_protos=['mqtt'],
#     id_=dai.device_addr,
#     idf_list=idf_list,
#     odf_list=odf_list,
#     name=dai.device_name,
#     profile={
#         'model': dai.device_model,
#         'u_name': dai.username,
#         'extra_setup_webpage': dai.extra_setup_webpage,
#         'device_webpage': dai.device_webpage,
#     },
#     register_callback=dai.register_callback,
#     on_register=dai.on_register,
#     on_deregister=dai.on_deregister,
#     on_connect=dai.on_connect,
#     on_disconnect=dai.on_disconnect
# )

def signal_handler(signal, frame):
    client.deregister()
    print(signal)
if hasattr(os.sys, 'winver'):
    signal.signal(signal.SIGBREAK, signal_handler)
else:
    signal.signal(signal.SIGTERM, signal_handler)

try:
    while True:
        for df_name in mpsa.idf_list:
            if not dai.flags.get(df_name) or dai.flags[df_name] == False:
                continue
            if not dai.device_features[df_name].push_data:
                continue
            _data = dai.device_features[df_name].push_data()
            if not isinstance(_data, NoData) and _data is not NoData:
                # print('push : ', _data)
                client.push(df_name, _data)
        time.sleep(1)
except KeyboardInterrupt:
    print("key interrupt")
finally:
    client.deregister()

