from os.path import join, exists
from numpy import arange


def list_time_dirs(case_dir):
    sys_dir = join(case_dir, 'system')
    with open(join(sys_dir, 'controlDict'), 'r') as handler:
        for ln in handler.readlines():
            ln2 = ln.split(' ')
            if ln2[0] == 'startTime':
                startTime = float(ln2[1][:-2])
            if ln2[0] == 'endTime':
                endTime = float(ln2[1][:-2])
            if ln2[0] == 'writeInterval':
                writeInterval = float(ln2[1][:-2])

    frames = arange(startTime, endTime, writeInterval) + writeInterval
    t_dirs = [f'{t:g}' for t in frames]

    return t_dirs

#


class readOFcase:
    def __init__(self, case_dir):
        self.case_dir = case_dir
        self.times = list_time_dirs(self.case_dir)
        self.existing = list(map(self.times, exists))
