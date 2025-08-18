#!/usr/bin/env bash

DOTS_OCR_SCRIPT="${DOTS_OCR_SCRIPT:-}"
DOTS_OCR_ARGS="${DOTS_OCR_ARGS:---use_hf true --num_thread 16}"


parse_pdf(){
    if [ -z "$DOTS_OCR_SCRIPT" ]; then
        echo "DOTS_OCR_SCRIPT is not set"
        exit 1
    fi

    if [ $# -ne 2 ]; then
        echo "Usage: $0 <pdf_file> <output_dir>"
        exit 1
    fi

    local pdf_file="$1"
    local output_dir="$2"

    local pdf_name=$(basename "$pdf_file")
    local pdf_name_without_ext="${pdf_name%.*}"
    local output_file="${output_dir}/${pdf_name_without_ext}.txt"

    if [ -f "$output_file" ]; then
        echo "Skipping $pdf_file because $output_file already exists"
        return
    fi

    python3 $DOTS_OCR_SCRIPT $DOTS_OCR_ARGS $pdf_file -o $output_file 
}

parse_pdf_dir(){
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <pdf_dir> <output_dir>"
        exit 1
    fi

    local pdf_dir="$1"
    local output_dir="$2"

    for pdf_file in $(find $pdf_dir -type f -name "*.pdf"); do
        parse_pdf $pdf_file $output_dir
    done
}

# local pdf_dir=""
# local output_dir=""

# 支持通过命令行参数覆盖
while [[ $# -gt 0 ]]; do
    case $1 in
        --script)
            DOTS_OCR_SCRIPT="$2"
            shift 2
            ;;
        --args)
            DOTS_OCR_ARGS="$2"
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

parse_pdf_dir $pdf_dir $output_dir