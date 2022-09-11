
def parseargs(cl_args):
    ops = {
        'case': cl_args[1],
        'interval': None,
        'time': None,
        'clean': False,
        'overwrite': False
    }

    for i in range(2, len(cl_args)):
        if '--range' == cl_args[i] and len(cl_args) >= i + 1:
            ops['interval'] = list(map(int, cl_args[i + 1].split(':')))
        if '--time' == cl_args[i] and len(cl_args) >= i + 1:
            ops['time'] = cl_args[i + 1]
        if '--clean' == cl_args[i]:
            ops['clean'] = True
        if '--overwrite' == cl_args[i]:
            ops['overwrite'] = True
        if '--old' == cl_args[i]:
            ops['oldstyle'] = True

    return ops
