one = []
two = []
swap = False
with open('out', 'r') as file:
	for line in file:
		if not swap:
			if not line.strip() == 'XXXXX':
				one += [line.strip()]
			else:
				swap = True
		else:
			two += [line.strip()]

print(one)
print(two)

count = 0
for i,j in enumerate(one):
	if j == two[i]:
		count += 1

print(count)
print(len(one))
print(count / len(one))
