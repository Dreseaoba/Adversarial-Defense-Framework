import os
import datetime
import yaml
import argparse

# from models.defenses.PerturbationInactivation import PerturbationInactivation
from models.defenses.get_defense import get_defense

from utils.logger import get_logger


def main(opt, logger):
    defense = opt.defenses[0]
    model = get_defense(defense['name'], *defense['args'], **defense['kwargs'])

    model_save_path = os.path.join(opt.save_path, 'ckpts')
    os.makedirs(model_save_path, exist_ok=True)

    try:
        model.train(
            logger, model_save_path, **opt.train_args)
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == '__main__':
    # utils.setup_seed(10)
    now = str(datetime.datetime.now())[:10] + "_" + str(datetime.datetime.now()).split()[1][:5]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=argparse.FileType(mode='r'), default='configs/deepfake_train.yml', help="configuration yml file")
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