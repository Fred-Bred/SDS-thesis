#!/bin/bash

source_dir="machamp/logs"
destination_dir="download"

find "$source_dir" -type f ! -name "*.pt" -exec cp --parents {} "$destination_dir" \;