import datetime
import time
import argparse
import yaml
import os

print('Importing...')

from utils.utils import AverageMeter, ETA
from utils.logger import get_logger
from evaluate import DeepfakeEvaluate, DeepfakeEvaluateTTA
from models.defenses.get_defense import get_defense

print('Starting...')


def main(opt, logger):
    # Prepairing data
    if 'b7' in opt.data_path:
        data_paths = [
            ['fake_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos", 1],
            ['fake_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos_attack_b7", 1],
            ['real_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos", 0],
            ['real_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos_attack_b7", 0]
        ]
    elif 'mpvit' in opt.data_path:
        data_paths = [
            # ['fake_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos", 1],
            ['fake_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos_attack_mpvit_c14", 1],
            # ['real_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos", 0],
            ['real_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos_attack_mpvit_c14", 0]
        ]
    else:
        data_paths = [
            # ['fake_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos", 1],
            ['fake_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos_attack_multi_c14", 1],
            # ['real_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos", 0],
            ['real_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos_attack_multi_c14", 0]
        ]
    # Prepairing models
    defenses = []
    for defense in opt.defenses:
        if defense['name']=='PIN':
            defense['args'] = [logger]
            defense['kwargs'] = {'mode': 'eval'}
        logger.info('loading defense model {} with args: {} {}'.format(defense['name'], defense['args'], defense['kwargs']))
        defenses.append(get_defense(defense['name'], *defense['args'], **defense['kwargs']))
    
    logger.info('loading dfd model')
    view_path = os.path.join(opt.save_path, 'samples')
    # view_path = None
    evaluater = DeepfakeEvaluateTTA(model=opt.model, defenses=defenses, logger=logger, batch_size=opt.batch_size, view_path=view_path)
    # starting
    for data_info in data_paths:
        files = os.listdir(data_info[1])[:100]
        total = len(files)
        logger.info('evaluating {}, file num: {}'.format(data_info[0], total))
        acc_count = 0
        avgMeter = AverageMeter()
        eta = ETA(total)
        for i, file in enumerate(files):
            print('\rProcessing {:0>4d}/{:0>4d} eta:{}'.format(i, total, eta()), end='  ')
            time_s = time.time()
            file_path = os.path.join(data_info[1], file)
            result, _ = evaluater(file_path)
            if result <= 0:
                total -= 1
            else:
                avgMeter.append(result)
                if data_info[2]==0 and result<0.5:
                    acc_count += 1
                elif data_info[2]==1 and result>0.5:
                    acc_count += 1
            eta.append(time.time() - time_s)
        print('')
        logger.info('{} result:\n Acc: {} ({}/{}), Avg output: {}'.format(data_info[0], acc_count/total, acc_count, total, avgMeter()))
            
            




if __name__ == '__main__':
    # utils.setup_seed(10)
    now = str(datetime.datetime.now())[:10] + "_" + str(datetime.datetime.now()).split()[1][:5]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=argparse.FileType(mode='r'), default='configs/deepfake.yml', help="configuration yml file")
    args = parser.parse_args()
    print(args)
    opt = yaml.load(args.config_file, Loader=yaml.FullLoader)
    opt['save_path'] = os.path.join(opt['save_path'], now + '_' + opt['exp_name'])
    os.makedirs(opt['save_path'], exist_ok=True)
    logger = get_logger(os.path.join(opt['save_path'], 'info.log'))
    yaml.dump(opt, open(os.path.join(opt['save_path'], 'config.yaml'), 'w'))
    opt = argparse.Namespace(**opt)
    logger.info(str(opt))

    main(opt, logger)

