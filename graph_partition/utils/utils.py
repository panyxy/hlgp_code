import numpy as np
import random
import torch
import time
import os, sys





class Logger(object):
    def __init__(self, file_name='logging.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, 'ab', buffering=0)
        self.log_disable = False
    def write(self, message):
        self.terminal.write(str(message))
        if not self.log_disable:
            self.log.write(str(message).encode('utf-8'))
    def flush(self):
        self.terminal.flush()
        if not self.log_disable:
            self.log.flush()
    def close(self):
        self.log.close()
    def disable_log(self):
        self.log_disable = True
        self.log.close()




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)



def create_logger(args, log_path, perm_model_type, problem_size, task_name, dist_type='uniform', part_model_type='gnn'):
    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    file_path = '{}_{}_{}/part-{}_perm-{}/{}_{}'.format(
        task_name, problem_size, dist_type, part_model_type,
        perm_model_type, time_stamp, round(np.random.rand(), 5)
    )

    args.file_path = os.path.join(log_path, file_path)
    args.file_name = os.path.join(log_path, file_path, 'log.txt')
    os.makedirs(args.file_path, exist_ok=True)

    args.results_dir = args.file_path

    sys.stdout = Logger(args.file_name, sys.stdout)
    sys.stderr = Logger(args.file_name, sys.stderr)

    return










