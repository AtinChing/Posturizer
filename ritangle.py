x = -10
max_y_sofar = 0
while x < 10:
    if x == 0: 
        x += 1
        continue
    a = ((17 * x/2) - (x** 2) - 1) / (x)  # Correct calculation
    #user_expression = f"(x**2)/(x**4 + {a}*(x**2) + 1)"
    y = (x**2)/(x**4 + a*(x**2) + 1)
    
    print(f"The value we get when original x = {x} and a = {a} is {y}")
    if max_y_sofar < y: 
        max_y_sofar = y
    x +=1