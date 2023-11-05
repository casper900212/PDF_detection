#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./extract-js.sh <INPUT_DIR> <OUTPUT_DIR>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

for FILENAME in "$INPUT_DIR"/*; do
    if [ -f "$FILENAME" ]; then  # Check if it's a regular file
        timeout 3s bash -c "
            echo \"$FILENAME\"
            NEW_FILENAME=\"$OUTPUT_DIR/$(basename "$FILENAME")\"

            RESULT=\$(peepdf -fl \"$FILENAME\" --command=\"extract js\")
            if [[ \$(echo \$RESULT) ]]; then
                RESULT=\$(echo \"\$RESULT\" | sed -e 's/\/\/ peepdf comment: Javascript code located in object [0-9]* (version [0-9]*) //g')
                echo \"\$RESULT\" > \"\$NEW_FILENAME\".tmp
                mv \"\$NEW_FILENAME\".tmp \"\$NEW_FILENAME\".js
            fi
        "

        if [ $? -eq 124 ]; then
            echo "Timeout: Extracting from $FILENAME took over 3 seconds."
        fi
    fi
done
