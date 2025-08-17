#!/usr/bin/env bash

set -x
set -e

DOTS_MERGE_SCRIPT="${DOTS_MERGE_SCRIPT:-merge_output.py}"
DOTS_MERGE_ARGS="${DOTS_MERGE_ARGS:---no-header-footer -c process}"


merge_pdf(){
    if [ -z "$DOTS_MERGE_SCRIPT" ]; then
        echo "DOTS_MERGE_SCRIPT is not set"
        exit 1
    fi

    if [ $# -ne 2 ]; then
        echo "Usage: $0 <ocr_result_file> <output_dir>"
        exit 1
    fi

    local pdf_file="$1"
    local output_dir="$2"

    local pdf_name
    pdf_name=$(basename "$pdf_file")
    local pdf_name_without_ext="${pdf_name%.*}"
    local final_output_dir="${output_dir}/${pdf_name_without_ext}"

    if [ -d "$final_output_dir" ]; then
        echo "Skipping $pdf_file because $final_output_dir already exists"
        return
    fi

    # 关键修改：所有路径都用双引号包围
    python3 "$DOTS_MERGE_SCRIPT" $DOTS_MERGE_ARGS -o "$final_output_dir" "$pdf_file"
}

merge_pdf_dir(){
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <pdf_dir> <output_dir>"
        exit 1
    fi

    local pdf_dir="$1"
    local output_dir="$2"

    find "$pdf_dir" -type f -name "*.jsonl" -print0 | while IFS= read -r -d '' pdf_file; do
        echo "Merging $pdf_file"
        merge_pdf "$pdf_file" "$output_dir"
    done
}

# local pdf_dir=""
# local output_dir=""

# 支持通过命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        --script)
            DOTS_MERGE_SCRIPT="$2"
            shift 2
            ;;
        --args)
            DOTS_MERGE_ARGS="$2"
            shift 2
            ;;
        *)
            if [ -z "$1" ]; then
                echo "Usage: $0 [--script <script_path>] [--args <args>] <pdf_dir> <output_dir>"
                exit 1
            fi
            pdf_dir="$1"
            output_dir="$2"
            shift 2
            ;;
    esac
done

merge_pdf_dir $pdf_dir $output_dir