import matplotlib.pyplot as plt

beta = []
helix = []
coil = []
overall = []

with open('out', 'r') as file:
	while file.readline():
		beta += [file.readline().strip()]
		helix += [file.readline().strip()]
		coil += [file.readline().strip()]
		overall += [file.readline().strip()]

beta = [float(x[18:]) * 100 for x in beta]
helix = [float(x[19:]) * 100 for x in helix]
coil = [float(x[18:]) * 100 for x in coil]
overall = [float(x[18:]) * 100 for x in overall]

a_beta = max(beta)
a_helix = max(helix)
a_coil = max(coil)
a_overall = max(overall)

idx = overall.index(a_overall)

print("Max beta accuracy: " + str(a_beta) + "%")
print("Max helix accuracy: " + str(a_helix) + "%")
print("Max coil accuracy: " + str(a_coil) + "%")
print("Max overall accuracy: " + str(a_overall) + "%")