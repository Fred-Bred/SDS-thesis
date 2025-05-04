parent_dirs=("logs/split1_150" "logs/split2_150" "logs/split3_150" "logs/split4_150" "logs/split5_150")

# Specific string to search for in file names
search_string="model_"

# Loop over all parent directories
for parent_dir in "${parent_dirs[@]}"; do
    # Extract the split number from the parent directory name
    split_num="${parent_dir:10:1}"
    
    # Get the latest subdirectory of the parent directory
    dir=$(ls -td "$parent_dir"/* | head -n1)
    
    # Check if it's a directory
    if [ -d "$dir" ]; then
        # Loop over all files in the subdirectory
        for file in "$dir"/*; do
            # Check if the file name contains the search string
            if [[ $file == *"$search_string"* ]]; then
                # Define the output directory
                output_dir="../Outputs/trained_models/k-folds/roberta-large_150"
                
                # Create the output directory if it doesn't exist
                mkdir -p "$output_dir"
                
                python3 predict.py "$file" "../Data/PACS_varying_lengths/test_combined_150.csv" "$output_dir/split${split_num}_test_preds.csv"
                break
            fi
        done
    fi
done