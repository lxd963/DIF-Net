class Config:
    """Configuration for heart sound classification"""
    
    def __init__(self):
        # Data settings
        self.data_root = 'dataset/PCG_and_STFT'
        self.image_size = 224
        self.batch_size = 64
        
        # Model settings
        self.model_name = 'dualpath_ffc_resnet18'
        self.num_classes = 2
        self.ratio = 0.75
        self.use_se = True
        self.use_att = True
        self.lfu = True
        
        # Training settings
        self.epochs = 100
        self.lr = 0.006
        self.weight_decay = 0.001
        self.gradient_clip = 1.0
        
        # Loss settings
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        
        # Cross-validation settings
        self.num_folds = 5
        self.cv_seed = 167
        
        # Hardware settings
        self.gpu_id = 0
        self.num_workers = 4
        
        # Output settings
        self.save_dir = 'outputs/cv_results'
        self.log_interval = 10
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}class Config:
