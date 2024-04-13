parent_dirs=("logs/split1_full" "logs/split2_full" "logs/split3_full" "logs/split4_full" "logs/split5_full")

# Specific string to search for in file names
search_string="model_"

# Loop over all parent directories
for parent_dir in "${parent_dirs[@]}"; do
    # Extract the split number from the parent directory name
    split_num="${parent_dir:10:1}"
    
    # Loop over all subdirectories of the parent directory
    for dir in "$parent_dir"/*; do
        # Check if it's a directory
        if [ -d "$dir" ]; then
            # Loop over all files in the subdirectory
            for file in "$dir"/*; do
                # Check if the file name contains the search string
                if [[ $file == *"$search_string"* ]]; then
                    # Define the output directory
                    output_dir="../Outputs/trained_models/k-folds/roberta-large_full"
                    
                    # Create the output directory if it doesn't exist
                    mkdir -p "$output_dir"
                    
                    python3 predict.py "$file" "../Data/k-folds/split${split_num}/val_PACS.csv" "$output_dir/split${split_num}_val_preds.csv"
                    break 2
                fi
            done
        fi
    done
done