#Nama : Bagas Syafiq Aero Pradana
#NIM : 21091397064/ 2021 B
# Multiple perceptron / Neuron batch and multiple layer 2

# inisialisasi numpy
import numpy as np

# inisialisasi variabel
# memasukan nilai variabel layer feature 10 dengan batch sejumlah 6
inputs = [
    [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 10.1, 10.3, 10.5],
    [3.1, 3.3, 3.5, 3.7, 3.9, 3.11, 3.13, 3.15, 3.17, 3.19],
    [5.1, 5.3, 5.5, 5.7, 5.9, 5.11, 5.13, 5.15, 5.17, 5.19],
    [2.2, 4.2, 2.4, 4.4, 2.6, 4.6, 2.8, 4.8, 2.10, 4.10],
    [6.1, 6.3, 6.5, 6.7, 6.9, 6.11, 6.13, 6.15, 6.17, 6.19],
    [10.2, 10.4, 10.6, 10.1, 10.3, 10.5, 10.7, 10.8, 10.9, 10.10],
]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5
weights1 = [
    [1.1, 1.3, 1.5, 1.7, 1.9, 1.2, 1.4, 1.6, 1.8, 1.10],
    [2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5],
    [4.1, 4.2, 4.3, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6],
    [6.3, 6.6, 6.1, 6.2, 6.4, 6.7, 8.1, 8.2, 8.3, 8.4],
    [20.1, 10.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1, 1.1],
]

# inisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu layer 1 = 5 neuron
biases1 = [3.2, 3.3, 3.4, 3.5, 3.1]

# inisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# memasukkan jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [
    [11.11, 11.2, 11.4, 12.5, 1.3],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [2.3, 5.6, 4.5, 6.7, 4.6]]

# inisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3
biases2 = [6.5, 6.7, 4.5]

transpose = np.dot(inputs, np.array(weights1).T)
print(transpose)
# output
# menghitung layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# menghitung layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print output layer2
print(layer2_outputs)
