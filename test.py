import os
import torch
import random
import numpy as np
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.metrics import t2v_metrics, v2t_metrics
from modules.loss import LossFactory
from trainer.trainer_stochastic import Trainer
from config.all_config import gen_log

# @WJM: solve num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # @WJM: add log
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # CLIP
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)

    # data I/O
    test_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)

    # metric
    if config.metric == 't2v':
        metrics = t2v_metrics
    elif config.metric == 'v2t':
        metrics = v2t_metrics
    else:
        raise NotImplemented

    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer(model=model,
                      loss=loss,
                      metrics=metrics,
                      optimizer=None,
                      config=config,
                      train_data_loader=None,
                      valid_data_loader=test_data_loader,
                      lr_scheduler=None,
                      writer=writer,
                      tokenizer=tokenizer)

    if config.load_epoch is not None:
        if config.load_epoch > 0:
            trainer.load_checkpoint("checkpoint-epoch{}.pth".format(config.load_epoch))
        else:
            trainer.load_checkpoint("model_best.pth")

    trainer.validate()


if __name__ == '__main__':
    main()

