python3 train.py --dataset_configs configs/split4/split4_50.json --retrain logs/dapt.config_pacs.0/2024.03.25_14.44.08/model_1.pt

python3 train.py --dataset_configs configs/split5/split5_50.json --retrain logs/dapt.config_pacs.0/2024.03.25_14.44.08/model_1.pt

./cv_predict_50.sh

./cv_full.sh

./cv_predict_full.sh