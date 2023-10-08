# Simple output parser
# Read through output.txt 
# Only retain the line starting with Epoch

import sys
import os
import re

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 outputparser.py <output.txt>")
        return

    output = sys.argv[1]
    if not os.path.exists(output):
        print("File does not exist")
        return

    with open(output, 'r') as f:
        lines = f.readlines()

    with open(output, 'w') as f:
        for line in lines:
            if re.match('Epoch', line):
                f.write(line)
    
if __name__ == '__main__':
    main()