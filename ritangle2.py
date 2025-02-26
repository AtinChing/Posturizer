a = 1
constant = 1
while constant < 2:
    constant += 0.001
    b = a * constant
    c = b * constant
    d = c * constant
    print(d/a)