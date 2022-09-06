#!/bin/env python

import sys
from numpy import load


def main():
    ifile = sys.argv[1]
    tmp = load(ifile)
    print(tmp)
    return None


if __name__ == '__main__':
    main()
