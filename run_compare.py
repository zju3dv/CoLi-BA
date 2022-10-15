import os
from argparse import ArgumentParser

seq_name_list = ['00','01','02','03','04','05','06','07','08','09','10',
                 'Alamo', 'Ellis_Island', 'Gendarmenmarkt',
                 'Madrid_Metropolis', 'Montreal_Notre_Dame', 'NYC_Library',
                 'Piazza_del_Popolo', 'Piccadilly', 'Roman_Forum',
                 'Tower_of_London', 'Trafalgar', 'Union_Square',
                 'Vienna_Cathedral', 'Yorkminster']

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='data_path') 
    parser.add_argument('--method', type=str, required=True,
                        help='method')
    return parser.parse_args()


if __name__ == '__main__':
    exe_path = './bin/main'
    args = get_opts()
    data_path = args.data_path
    method = args.method

    if not data_path.endswith('/'):
        data_path = data_path+'/'

    for seq_name in seq_name_list:
        print(seq_name)  
        seq_path = data_path+seq_name+'/'
        os.system(exe_path+' '+seq_path+' '+method) 
