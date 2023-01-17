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

import ffmpeg

print('Starting...')

# video_pathname:待压缩视频路径
# compress_pathname:压缩后视频保存路径


def ffmepg_compress(video_pathname, video_filename, compress_pathname, crf=34):
    probe = ffmpeg.probe(video_pathname)
    format = probe['format']
    bit_rate = format['bit_rate']
    save_path = os.path.join(compress_pathname, video_filename[:-4] + '_crf' + str(crf) + '.mp4')
    r_framerate = probe['streams'][0]['r_frame_rate']
    cmd = r"ffmpeg -i {} -framerate {} -crf {} -c:v libx264 -b:v {} -y {}"
    cmd = cmd.format(video_pathname, 
                    r_framerate, crf, bit_rate, 
                    save_path)
    os.system(cmd)
    return save_path
    


def main(opt, logger):
    # Prepairing data
    compress_pathname = '/data/guohaoxun/workspace/atk/AdvDefenseFramework/datasets'
    compress_pathname = os.path.join(compress_pathname, 'compressed_videos')
    os.makedirs(compress_pathname, exist_ok=True)

    if 'b7' in opt.data_path:
        data_paths = [
            ['fake_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos", 1],
            ['real_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos", 0],
            # ['fake_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/manipulated_sequences/Deepfakes/c23/videos_attack_b7", 1],
            # ['real_adv_path', "/data/fanglingfei/dataset/faceforensics_c23/original_sequences/youtube/c23/videos_attack_b7", 0]
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
    # for defense in opt.defenses:
    #     if defense['name']=='PIN':
    #         defense['args'] = [logger]
    #         defense['kwargs'] = {'mode': 'eval'}
    #     logger.info('loading defense model {} with args: {} {}'.format(defense['name'], defense['args'], defense['kwargs']))
    #     defenses.append(get_defense(defense['name'], *defense['args'], **defense['kwargs']))
    
    logger.info('loading dfd model')
    view_path = os.path.join(opt.save_path, 'samples')
    # view_path = None
    evaluater = DeepfakeEvaluateTTA(model=opt.model, defenses=defenses, logger=logger, batch_size=opt.batch_size, view_path=view_path)
    # starting
    for data_info in data_paths:
        files = os.listdir(data_info[1])[:20]
        total = len(files)
        logger.info('evaluating {}, file num: {}'.format(data_info[0], total))
        acc_count = 0
        avgMeter = AverageMeter()
        eta = ETA(total)
        for i, file in enumerate(files):
            print('\rProcessing {:0>4d}/{:0>4d} eta:{}'.format(i, total, eta()), end='  ')
            time_s = time.time()
            file_path = os.path.join(data_info[1], file)
            compress_path = ffmepg_compress(file_path, file, compress_pathname, 40)
            # run
            result, _ = evaluater(compress_path)
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

