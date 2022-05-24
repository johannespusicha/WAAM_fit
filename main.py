import sys
import WAAMEvaluator


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print('Please provide a path for the step file to process.\n')
        quit(0)
    InputPath = args[1]
    OutputPath = 'out.txt'
    option = ''
    for i in range(2, len(args)):
        if option == '':
            option = args[i]
        else:
            if option == '-o':
                OutputPath = args[i]
            option = ''
    WAAMEvaluator.evaluate(InputPath, OutputPath)
