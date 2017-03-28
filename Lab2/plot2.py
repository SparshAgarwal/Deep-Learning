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

e = [x for x in range(0, 200)]

train[0] = [float(x[18:]) * 100 for x in train[0][:200]]
tune[0] = [float(x[18:]) * 100 for x in tune[0][:200]]
test[0] = [float(x[18:]) * 100 for x in test[0][:200]]
train[1] = [float(x[19:]) * 100 for x in train[1][:200]]
tune[1] = [float(x[19:]) * 100 for x in tune[1][:200]]
test[1] = [float(x[19:]) * 100 for x in test[1][:200]]
train[2] = [float(x[18:]) * 100 for x in train[2][:200]]
tune[2] = [float(x[18:]) * 100 for x in tune[2][:200]]
test[2] = [float(x[18:]) * 100 for x in test[2][:200]]
train[3] = [float(x[18:]) * 100 for x in train[3][:200]]
tune[3] = [float(x[18:]) * 100 for x in tune[3][:200]]
test[3] = [float(x[18:]) * 100 for x in test[3][:200]]

plt.title('Overall Accuracy versus Epoch(time)')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch / Time')

t1, = plt.plot(e, train[3], 'red', label='Training Set')
t2, = plt.plot(e, tune[3], 'blue', label='Tuning Set')
t3, = plt.plot(e, test[3], 'green', label='Testing Set')

plt.legend(handles=[t1, t2, t3])

plt.show()