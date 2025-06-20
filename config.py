mt5_path = "./pretrained_weight/mt5-base"

# label paths
train_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.train",
                    "WLASL": "./data/WLASL/labels-2000.train"
                    }

dev_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.dev",
                    "WLASL": "./data/WLASL/labels-2000.dev"
                    }

test_label_paths = {
                    "CSL_News": "/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/data/train/CSL_News_Labels.json",
                    "CSL_Daily": "./data/CSL_Daily/labels.test",
                    "WLASL": "./data/WLASL/labels-2000.test"
                    }


# video paths
rgb_dirs = {
            "CSL_News": '/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/rgb_format',
            "CSL_Daily": './dataset/CSL_Daily/sentence-crop',
            "WLASL": "./dataset/WLASL/rgb_format"
            }

# pose paths
pose_dirs = {
            "CSL_News": '/mnt/fast/nobackup/scratch4weeks/ef0036/cslnews/rgb_format',
            "CSL_Daily": '/mnt/fast/nobackup/scratch4weeks/ef0036/csl_daily/pose_format',
            "WLASL": "/mnt/fast/nobackup/scratch4weeks/ef0036/wlasl"
            }
