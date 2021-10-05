class Config(object):
    def __init__(self, config):
        
        self.dataset = ''
        self.output_dir = ''
        self.run_name = 'run'

        self.model = 'gcn'
        self.layer = 'gconv'
        self.activation = 'relu'
        self.input_norm = False
        self.layer_norm = False
        self.num_layers = 2
        self.hidden_size = 16
        
        # g for gconv, l for linear, a for attention, s for sageconv
        self.arch = 'gg'
        self.residual = False


        self.num_samplers = 5
        self.sampler = 'subgraph'
        self.num_neighbors = [10,10]
        self.minibatch = 'random'
        self.minibatch_size = 256
        self.local_updates = 5

        # Correction server settings
        self.server_sampler = 'subgraph'
        self.server_num_neighbors = [10,10]
        self.server_minibatch = 'random'
        self.server_minibatch_size = 256
        self.server_updates = 1
        self.server_lr = 1e-3
        self.server_start_epoch = 0
        self.server_opt_sync = False
        self.rho = 1
        self.inc_k = False

        self.loss = 'xentropy'
        self.optim = 'adam'
        self.lr = 2e-2
        self.dropout = 0
        self.wd = 0
        self.num_epochs = 200
        self.val_patience = 2
        self.val_step = 1

        self.gpu = 0
        self.cpu = False
        self.cpu_val = False
        self.num_gpus = 4
        
        self.part_method = 'random'
        self.part_args = ''
        self.num_workers = 2

        # Mostly unused!
        self.hist_period = 0
        self.hist_exp = False
        self.stratified = False
        self.sync_local = True
        self.full_correct = False
        self.use_sampling = False
        self.weight_avg = True
        
        for key, value in config.items():
            setattr(self, key, value)

        if self.dataset is not None and type(self.dataset) != str:
            self.dataset = self.dataset.name

    
    def __repr__(self):
        all_config = vars(self)
        for c in all_config:
            print(c, all_config[c])
        
        return ""

    @property
    def world_size(self):
        return self.num_workers

    @property
    def partitioned_dir(self):
        # TOOD: better handling
        if self.part_method == 'overhead':
            return f'partitioned/overhead-{self.part_args}-{self.num_workers}'
        else:
            return f'partitioned/{self.part_method}-{self.num_workers}'

    @property
    def processed_filename(self):
        # if self.dataset in ['proteins', 'arxiv', 'products']:
        #     return 'processed/geometric_data_processed.pt'
        # else:
        #     return 'processed/data.pt'
        return 'processed/data.pt'