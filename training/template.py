from os.path import exists
from pathlib import  Path
FILE_NAME = 'template.in'
LOCAL_FOLDER_NAME = 'template'
LOCAL_DATA_ROOT_FOLDER_NAME = './data'


def read_input():
    data_path = Path(LOCAL_DATA_ROOT_FOLDER_NAME) / LOCAL_FOLDER_NAME
    on_server = exists(FILE_NAME)
    base_path = Path('.') if on_server else data_path
    with open(base_path / FILE_NAME, 'r') as f:
        result = f.read()
    return result


def optimize():
    result = read_input()
    print(f'{result}')


def main():
    optimize()


if __name__ == '__main__':
    main()
