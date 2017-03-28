import matplotlib.pyplot as plt

train = [[],[],[],[]]
tune = [[],[],[],[]]
test = [[],[],[],[]]

with open('out.data', 'r') as file:
	while file.readline():
		train[0] += [file.readline().strip()]
		train[1] += [file.readline().strip()]
		train[2] += [file.readline().strip()]
		train[3] += [file.readline().strip()]
		file.readline()
		tune[0] += [file.readline().strip()]
		tune[1] += [file.readline().strip()]
		tune[2] += [file.readline().strip()]
		tune[3] += [file.readline().strip()]
		file.readline()
		test[0] += [file.readline().strip()]
		test[1] += [file.readline().strip()]
		test[2] += [file.readline().strip()]
		test[3] += [file.readline().strip()]

e = [x for x in range(0, 100)]

train[0] = [float(x[18:]) * 100 for x in train[0][:100]]
tune[0] = [float(x[18:]) * 100 for x in tune[0][:100]]
test[0] = [float(x[18:]) * 100 for x in test[0][:100]]
train[1] = [float(x[19:]) * 100 for x in train[1][:100]]
tune[1] = [float(x[19:]) * 100 for x in tune[1][:100]]
test[1] = [float(x[19:]) * 100 for x in test[1][:100]]
train[2] = [float(x[18:]) * 100 for x in train[2][:100]]
tune[2] = [float(x[18:]) * 100 for x in tune[2][:100]]
test[2] = [float(x[18:]) * 100 for x in test[2][:100]]
train[3] = [float(x[18:]) * 100 for x in train[3][:100]]
tune[3] = [float(x[18:]) * 100 for x in tune[3][:100]]
test[3] = [float(x[18:]) * 100 for x in test[3][:100]]

plt.title('Overall Accuracy versus Epoch(time)')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch / Time')

t1, = plt.plot(e, test[0], 'red', label='Beta')
t2, = plt.plot(e, test[1], 'blue', label='Helix')
t3, = plt.plot(e, test[2], 'green', label='Coil')
t4, = plt.plot(e, test[3], 'black', label='Overall')

plt.legend(handles=[t1, t2, t3, t4])

plt.show()