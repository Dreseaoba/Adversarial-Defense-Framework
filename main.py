import datetime
import argparse
import yaml
import os

print('Importing...')

import utils.utils as utils
from utils.logger import get_logger
from utils.data_loader import ImagePairSet
from models.face_recognition.FaceRecognition import FaceRecognizer
from models.defenses.get_defense import get_defense

print('Starting...')


def main(opt):
    # Prepairing data
    logger.info('loading data from {}'.format(opt.data_path))
    dataset = ImagePairSet(opt.data_path, targeted=False)
    src, adv = dataset[0]
    # Prepairing models
    logger.info('loading FR model {}'.format(opt.fr_model))
    face_recognizer = FaceRecognizer(opt.fr_model, logger)
    logger.info('loading defense model {}'.format(opt.defense))
    defense = get_defense(opt.defense)
    # No defense
    face_recognizer([(src, adv, True)])
    # Do defense
    adv_df = defense(adv[None, :])
    face_recognizer([(src, adv_df[0], True)])




if __name__ == '__main__':
    # utils.setup_seed(10)
    now = str(datetime.datetime.now())[:10] + "_" + str(datetime.datetime.now()).split()[1][:5]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=argparse.FileType(mode='r'), default='config.yml', help="configuration yml file")
    args = parser.parse_args()
    print(args)
    opt = yaml.load(args.config_file, Loader=yaml.FullLoader)
    # opt['save_path'] = os.path.join(opt['save_path'], now)
    os.makedirs(opt['save_path'], exist_ok=True)
    logger = get_logger(os.path.join(opt['save_path'], 'info.log'))
    yaml.dump(opt, open(os.path.join(opt['save_path'], 'config.yaml'), 'w'))
    opt = argparse.Namespace(**opt)
    
    logger.info(str(opt))
    main(opt)

