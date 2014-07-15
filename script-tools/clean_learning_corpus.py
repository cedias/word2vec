import sys
from num2words import num2words
import re


def cleanfile(filename, output, stats=True):

    if stats:
        num_lines = sum(1 for line in open(filename))
        i = 0

    fin = open(filename, "r")
    fou = open(output, "w")
    delChars = [".", "?", "!", ",", "\"", "(", ")", "{", "}", ":", ";", "#", "*", "/", "\\", "'", "-"]

    for line in fin:
        if len(line) > 1:
            for char in delChars:
                line = line.replace(char, "")
            line = line.replace("&", "and")

            nums = re.findall(r'\d+', line)
            for num in nums:
                o = num
                n = str(" "+num2words(int(num))+" ")

                line = line.replace(o, n)
            fou.write(line)

        if stats:
            i += 1
            percent = i/float(num_lines)*100
            sys.stdout.write("\r (%d/%d)  %d%%  " % (i, num_lines, percent))
            sys.stdout.flush()
    print "\n"

cleanfile(sys.argv[1], sys.argv[2])
