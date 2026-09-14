#!/bin/bash
# Remove generated outputs — preserves mesh cache

echo "Cleaning outputs..."
rm -rf outputs/
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "Done. cache/ preserved (mesh stays)."
echo "To force mesh regeneration: rm -rf cache/"
