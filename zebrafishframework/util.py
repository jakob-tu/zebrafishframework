import os
import pickle
import random
import string
import subprocess
import sys


class ExecutionException(Exception):
    pass


__PATH_ADDONS = [
    '/Users/koesterlab/bin/ants/bin',
    '/opt/local/bin'
]


def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def call(cmd, print_output=False):
    """
    Sensible wrapper function to call something as if one did it in a terminal
    :param cmd:
    :param return_output Capture output and return (do not print)
    :return: command stdout
    """

    newenv = os.environ
    newenv['PATH'] += ':' + ':'.join(__PATH_ADDONS)

    kwargs = dict()
    kwargs['stderr'] = subprocess.PIPE
    kwargs['stdout'] = subprocess.PIPE

    out = ''
    p = subprocess.Popen(cmd, shell=True, env=newenv, **kwargs)
    for line in p.stdout:
        if print_output:
            sys.stdout.write(str(line, 'utf-8'))
            sys.stdout.flush()
        out += str(line, 'utf-8')

    if p.returncode:
        raise ExecutionException(p.stderr.readlines())

    return out


def format_time(e):
    e = int(e)
    return '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)


def shutdown():
    os.system('sudo shutdown -h now')


def await_save(ar, fn):
    with open(fn, 'wb') as f:
        ar.wait()
        pickle.dump(ar.get(), ar.error, ar.serial_time, ar.wall_time)


def await_save_shutdown(ar, fn):
    await_save(ar, fn)
    shutdown()