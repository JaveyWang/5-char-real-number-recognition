def _int2char(int_):
    int2char = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
    return int2char[int_]

def intlist2str(intlist):
    return "".join([_int2char(int_) for int_ in intlist])