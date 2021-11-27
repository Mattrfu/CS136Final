import matplotlib.pyplot as plt
f = open("results1.txt", "r")
myArray = []
line = f.readline()
y = []
while line:
    line.replace("\n", "")
    myArray += [line]
    if len(myArray) > 100:
        del myArray[0]
    if len(myArray) == 100:
        num = myArray.count("True\n")/100.0
        y += [num]
    line = f.readline()
plt.plot(y)
plt.show()

