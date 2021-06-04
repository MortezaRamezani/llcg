
import os
import time
import yaml
import copy
import numpy as np

from multiprocessing import Value, Array, Manager

class Stats(object):
    
    def __init__(self, config):


        self.config = config

        # Loss and Scores
        # manager = Manager()
        # self.train_loss = manager.list()
        # self.train_scores = manager.list()
        # self.val_loss = manager.list()
        # self.val_scores = manager.list()
        # self.test_score = manager.Value('d', 0)

        self.train_loss = []
        self.train_scores = []
        self.val_loss = []
        self.val_scores = []
        self.test_score = []

        # best model
        self.best_model = []
        self.best_val_score = 0
        self.best_val_loss = 1e10
        self.best_val_epoch = 0

        self.best_val_buff = []

        # TODO
        # Timing info
        self.train_time = []
        self.val_time = []
        self.test_time = 0
        self.comm_cost = []
        

    @property
    def run_id(self):
        current_counter = 1

        if os.path.exists(self.config.output_dir):
            for fn in os.listdir(self.config.output_dir):
                if fn.startswith(self.config.run_name+'-') and fn.endswith('npz'):
                    current_counter += 1
                
        return '{}-{:03d}'.format(self.config.run_name, current_counter)

    @property
    def run_output(self):
        output = os.path.join(self.config.output_dir, self.run_id)
        return output

    
    def save(self):

        if self.config.output_dir == '':
            return None, None
            
        config_vars = vars(self.config)

        stats_vars = copy.copy(vars(self))
        stats_vars.pop('config', None)

        # create output folder
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        
        # save model to torch TODO: later
        # remove from stats
        stats_vars.pop('best_model', None)

        # print(stats_vars)
        
        # save config and stats to npy
        np.savez(self.run_output, config=config_vars, stats=stats_vars)
    
        return config_vars, stats_vars

    @staticmethod
    def load(stats_file):
        all_data = np.load(stats_file, allow_pickle=True)
        config = all_data['config'][()]
        stats = all_data['stats'][()]
        return config, stats
