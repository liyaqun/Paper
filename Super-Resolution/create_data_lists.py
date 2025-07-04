from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_folders=['D:/OneDrive/桌面/ML/data/train/train2017',
                                     'D:/OneDrive/桌面/ML/data/valid/val2017'],
                      test_folders=['D:/OneDrive/桌面/ML/data/Set5',
                                    'D:/OneDrive/桌面/ML/data/Set14',
                                    'D:/OneDrive/桌面/ML/data/BSD100'],
                      min_size=100,
                      output_folder='./')
