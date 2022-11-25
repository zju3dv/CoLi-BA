import os
from argparse import ArgumentParser

seq_name_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                 'Alamo', 'Ellis_Island', 'Gendarmenmarkt',
                 'Madrid_Metropolis', 'Montreal_Notre_Dame', 'NYC_Library',
                 'Piazza_del_Popolo', 'Piccadilly', 'Roman_Forum',
                 'Tower_of_London', 'Trafalgar', 'Union_Square',
                 'Vienna_Cathedral', 'Yorkminster']

# The different noises are only used to keep the initial average reprojection-error at about 10px
noise_list = [5e-4, 5e-4, 2e-4, 5e-4, 1e-3, 1e-3, 1e-3, 1e-3, 5e-4, 2e-4,5e-4,
            2e-3, 2e-3, 4e-3, 
            2e-3, 2e-3, 2e-3, 
            2e-3, 1e-3, 4e-3,
            2e-3, 2e-3, 1e-3, 
            4e-3, 4e-3]


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

    for i in range(len(seq_name_list)):
        seq_name = seq_name_list[i]
        noise = str(noise_list[i])
        print('sequence:{} noise:{}'.format(seq_name,noise))
        seq_path = data_path+seq_name+'/'
        os.system(exe_path+' '+method+' '+seq_path+' '+noise)
