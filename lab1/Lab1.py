import numpy as np
import math
import random
array_x = np.random.randint(1, 21, (8,3))  #заповнюємо матрицю планування випадковими числами від 1 до 20 включно
a_0, a_1, a_2, a_3 = random.randint(5, 25), random.randint(5, 25), random.randint(5, 25), random.randint(5, 25) #коефіцієнти а в межах від 5 до 25
min_xi = np.amin(array_x, axis=0)  #знаходимо Хmin
max_xi = np.amax(array_x, axis=0)   #знаходимо Хmax
b = np.concatenate((array_x, np.zeros([2, 3])))
array = np.concatenate((b, np.zeros([10, 4])), axis=1)    #робимо матрицю розмірністю 10 на 7
for i in range(3):
     array[8][i] = (max_xi[i] + min_xi[i]) / 2   #x0
     array[9][i] = array[8][i] - min_xi[i]       #dx

for i in range(8):
    array[i][3] = a_0 + a_1 * array[i][0] + a_2 * array[i][1] + a_3 * array[i][2]   #Yi

for i in range(8):
    array[i][4] = ((array[i][0] - array[8][0]) / (array[9][0]))   #Xн1, Xн2, Xн3
    array[i][5] = ((array[i][1] - array[8][1]) / (array[9][1]))
    array[i][6] = ((array[i][2] - array[8][2]) / (array[9][2]))

y_etalon = a_0+a_1*array[8][0]+a_2*array[8][1]+a_3*array[8][2]
print("Y_etalon = ", y_etalon)
Y = []
for i in range(8):
    Y.append(math.pow((array[i][3]-y_etalon), 2))
answer = max(Y)
print("max((Y - Y_elalon)^2) = ", answer)

print("\tX1      X2      X3      Y       Xн1      Xн2      Xн3")
items = ["1", "2", "3", "4", "5", "6", "7", "8", "x0", "dx"]
for i in range(10):
    print(items[i], end="\t")
    for j  in range(7):
        if (i>=8 and j>=3):
            print("  -  ", end="   ")
        else:
            print("{:>5.2f}".format(array[i][j]), end="   ")
    print()

