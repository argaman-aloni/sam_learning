import pathlib
import signal
import subprocess


def sigint_handler(sig, frame):
    signal.signal(signal.SIGINT, sigint_handler)
    print('\nCtrl-C pressed. Do you want to quit? (y/n): ', end="")
    response = input().strip().lower()
    if response in ['y', 'yes']:
        pathlib.Path('temp.sh').unlink()
        exit(0)


def main():
    signal.signal(signal.SIGINT, sigint_handler)
    for id in range(10535770, 10535789):
        subprocess.check_output(['scancel', str(id)])
        print(f"cancelled job {id}")


if __name__ == '__main__':
    main()
