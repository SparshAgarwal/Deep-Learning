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

a_beta = sum(beta) / len(beta)
a_helix = sum(helix) / len(helix)
a_coil = sum(coil) / len(coil)
a_overall = sum(overall) / len(overall)

print("Average beta accuracy: " + str(a_beta) + "%")
print("Average helix accuracy: " + str(a_helix) + "%")
print("Average coil accuracy: " + str(a_coil) + "%")
print("Average overall accuracy: " + str(a_overall) + "%")
