#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
@Project    ：MGFmiRNALOC: mRNA subcellular localization prediction and analysis by exploiting molecular graph feature and CBAM.
@Description: 10-fold cross-validation of MGFmiRNALOC.
'''
print(__doc__)

import sys, argparse


def main():

    if not os.path.exists(args.output):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.output)
    if not os.path.exists(args.positive) or not os.path.exists(args.negative):
        print("The input data not exist! Error\n")
        sys.exit()

    funciton(args.positive, args.negative, args.output, args.fold)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Manual to the MGFmiRNALOC')
    parser.add_argument('-p', '--positive', type=str, help='positive data')
    parser.add_argument('-n', '--negative', type=str, help='negative data')
    parser.add_argument('-f', '--fold', type=int, help='k-fold cross-validation', default=10)
    parser.add_argument('-o', '--output', type=str, help='output folder')
    args = parser.parse_args()

    from train import *
    main()
#python ./code/main.py -p ./data/Positive.txt -n ./data/Negative.txt -f 10 -o ./result/