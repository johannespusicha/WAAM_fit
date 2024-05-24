import argparse
import sys
from waam_fit import evaluate

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("-o", type=str)
    parser.add_argument("-s", type=float)
    parser.add_argument("-b", type=float, nargs=6)
    
    args = parser.parse_args()

    output_path = args.o if args.o else "./"
    step_size = args.s if args.s else 0.0
    args.b = args.b if args.b else []
    base_points = (((args.b[0], args.b[1], args.b[2]), (args.b[3], args.b[4], args.b[5])) 
                   if len(args.b) == 6 else None)
    evaluate.evaluateGeometry(args.input_path, output_path, step_size, base_points)

if __name__ == '__main__':
    main()