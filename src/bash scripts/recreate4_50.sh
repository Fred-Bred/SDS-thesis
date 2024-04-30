# # Pre-trained mental-roberta
# python3 train.py --dataset_configs configs/split4/split4_50.json --retrain logs/dapt.config_pacs.0/2024.03.25_14.44.08/model_1.pt

# ./cv_predict_50.sh

# # Clean mental-roberta
# python3 train.py --dataset_configs configs/split4/split4_50.json

# parent_dir="logs/split4_50"
# search_string="model_"

# # Get the latest subdirectory of the parent directory
#     dir=$(ls -td "$parent_dir"/* | head -n1)
    
#     # Check if it's a directory
#     if [ -d "$dir" ]; then
#         # Loop over all files in the subdirectory
#         for file in "$dir"/*; do
#             # Check if the file name contains the search string
#             if [[ $file == *"$search_string"* ]]; then
#                 # Define the output directory
#                 output_dir="../Outputs/trained_models/k-folds/MentalRoBERTa_50"
                
#                 # Create the output directory if it doesn't exist
#                 mkdir -p "$output_dir"
                
#                 python3 predict.py "$file" "../Data/k-folds/split${split_num}/val_50.csv" "$output_dir/split4_val_preds.csv"
#                 break
#             fi
#         done
#     fi

# # pre-trained roberta
# python3 train.py --dataset_configs configs/split4/split4_50.json --retrain logs/dapt/2024.03.22_12.54.40/model_1.pt --parameters_config configs/params_roberta.json

# Predict for pre-trained roberta
# Get the latest subdirectory of the parent directory
    # dir=$(ls -td "$parent_dir"/* | head -n1)
    
    # # Check if it's a directory
    # if [ -d "$dir" ]; then
    #     # Loop over all files in the subdirectory
    #     for file in "$dir"/*; do
    #         # Check if the file name contains the search string
    #         if [[ $file == *"$search_string"* ]]; then
    #             # Define the output directory
    #             output_dir="../Outputs/trained_models/k-folds/pretrained_roberta_50"
                
    #             # Create the output directory if it doesn't exist
    #             mkdir -p "$output_dir"
                
    #             python3 predict.py "$file" "../Data/k-folds/split${split_num}/val_50.csv" "$output_dir/split4_val_preds.csv"
    #             break
    #         fi
    #     done
    # fi

# Clean roberta
python3 train.py --dataset_configs configs/split4/split4_50.json --parameters_config configs/params_roberta.json

# Get the latest subdirectory of the parent directory
    dir=$(ls -td "$parent_dir"/* | head -n1)
    
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Loop over all files in the subdirectory
        for file in "$dir"/*; do
            # Check if the file name contains the search string
            if [[ $file == *"$search_string"* ]]; then
                # Define the output directory
                output_dir="../Outputs/trained_models/k-folds/roberta-base_50"
                
                # Create the output directory if it doesn't exist
                mkdir -p "$output_dir"
                
                python3 predict.py "$file" "../Data/k-folds/split4/val_50.csv" "$output_dir/split4_val_preds.csv"
                break
            fi
        done
    fi

# roberta-large
python3 train.py --dataset_configs configs/split4/split4_50.json --parameters_config configs/params_roberta_large.json

# Get the latest subdirectory of the parent directory
    dir=$(ls -td "$parent_dir"/* | head -n1)
    
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Loop over all files in the subdirectory
        for file in "$dir"/*; do
            # Check if the file name contains the search string
            if [[ $file == *"$search_string"* ]]; then
                # Define the output directory
                output_dir="../Outputs/trained_models/k-folds/roberta-large_50"
                
                # Create the output directory if it doesn't exist
                mkdir -p "$output_dir"
                
                python3 predict.py "$file" "../Data/k-folds/split4/val_50.csv" "$output_dir/split4_val_preds.csv"
                break
            fi
        done
    fi