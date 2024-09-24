import subprocess
from config import PORT_TENSORBOARD_SERVER, PROJECT

command = [
    'tensorboard',
    '--logdir', f'{PROJECT}',
    '--host', '0.0.0.0',
    '--port', f'{PORT_TENSORBOARD_SERVER}'
]

process = subprocess.run(command)

