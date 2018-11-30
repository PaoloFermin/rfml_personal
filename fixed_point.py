import numpy as np
from struct import *
import math

'''
Takes in a floating point number and converts it into a byte list (same format as when you unpack a value into 4 bytes)
that uses fixed point format instead of the IEEE format
1st bit is the signed bit
2nd bit is the whole number bit, 0 or 1, since we are only dealing with a range of -1.999... to 1.999....
Bits 3 - 32 represent the decimal
'''
def to_fixed_point(value) :
    sign = 0
    if value < 0:
        sign = 1
        value = value * -1

    whole_number = int(math.floor(value))
    decimal = value - whole_number

    binArray = []
    for x in range(30):
        decimal = decimal * 2
        binArray.append(int(math.floor(decimal)))
        if decimal >= 1:
            decimal = decimal - 1

    byte3 = sign;
    byte3 = (byte3 << 1) + whole_number
    for x in range(6):
        byte3 = (byte3 << 1) + binArray[x]

    byte2 = 0;
    for x in range(8):
        byte2 = (byte2 << 1) + binArray[x + 6]

    byte1 = 0;
    for x in range(8):
        byte1 = (byte1 << 1) + binArray[x + 14]

    byte0 = 0;
    for x in range(8):
        byte0 = (byte0 << 1) + binArray[x + 22]

    byteList = [byte0, byte1, byte2, byte3]
    return byteList


'''
Takes in a byte list (same format as when you unpack a value into 4 bytes) in our fixed point notion
and converts it back into a float
'''
def to_floating_point(byteList):
    # Gets the signed and whole number bit
    sign_mask = 1 << 7
    whole_mask = 1 << 6
    sign = (sign_mask & byteList[3]) >> 7
    whole_number = (whole_mask & byteList[3]) >> 6

    # Calculates the decimal number from the remaining bits
    decimal_number = 0;
    for x in reversed(range(30)):
        byte = int(math.floor(x / 8))
        bit_in_byte = (x % 8)
        mask = 1 << bit_in_byte
        value = (mask & byteList[byte]) >> bit_in_byte
        # print("byte: " + str(byte) + " bit: " + str(bit_in_byte) + " value: " + str(value))
        decimal_number = decimal_number + (value * (2 ** ((30 - x) * -1)))

    number = whole_number + decimal_number
    if sign == 1:
        number = number * -1

    return number

def byte_list_to_bin(byteList):
    byte0 = format(byteList[0], 'b')
    byte1 = format(byteList[1], 'b')
    byte2 = format(byteList[2], 'b')
    byte3 = format(byteList[3], 'b')
    bin = pad_zeros(byte3) + pad_zeros(byte2) + pad_zeros(byte1) + pad_zeros(byte0)
    return bin


def pad_zeros(byte):
    padding = ""
    for x in range(8 - len(byte)):
        padding = padding + "0"
    return padding + byte;
