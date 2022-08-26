from train import train_main
from test import test_main
from sys import argv
from gc import collect

def main(device: str, small_: str, run: str):
    small = True if small_ == 'small' else False
    print(f'aasu: using device {device}')
    print(f'aasu: using {"small" if small else "large"} datasets')
    try:
        collect()
        if 'train' in run:
            train_main(device, small)
        if 'test' in run:
            test_main(device, small)
    except KeyboardInterrupt:
        print('aasu: interrupted')

if __name__ == '__main__':
    main(argv[1], argv[2], argv[3])