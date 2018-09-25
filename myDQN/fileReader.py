import matplotlib.pyplot as plt


data = []

with open("rewards.txt") as file:
    for line in file:
        arr = []
        for word in line.split():
            arr.append(word)
        for words in arr:
            if words == "episode":
                data.append(arr)

actual_data = []

for array in data:
    a = []
    for i,word in enumerate(array):
        if(word == "episode"):
            a.append(array[i+1])
            a.append(array[i+2])
    actual_data.append(a)

x_data = []
y_data = []

for array in actual_data:
    for i,number in enumerate(array):
        if i == 0:
            x_data.append(int(number))
        if i == 1:
            y_data.append(float(number))

plt.plot(x_data, y_data)
plt.show()


