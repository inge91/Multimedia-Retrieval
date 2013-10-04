# Calculates the distance between each queryset and each 
import os
import shutil
import math

# the name of the files that should be copied end with this string
query_dir = '.\\QuerySet\\'

# Destination for copied files
test_dir = ".\\Testset\\"
 
# Result file directory
result_dir = ".\\Results\\" 

def file_to_list(f):
	#loop thorugh all
	values = f.readline()
	p = values.split('\t')
	# Remove the newline
	p = p[:-1]
	p = map(float, p)
	return p


def evaluate_distances(q_file):
	q_vector = file_to_list(q_file)
	name = q_file.name[11:14]
	print q_file.name
	print name
	# create file containing ordered list of file rankings
	result_file = open(result_dir + name + '.txt', 'w+')
	
	#list of distances
	distances = []
	
	for filename in os.listdir(test_dir):
		t_file = open(test_dir + filename, 'r')
		t_vector = file_to_list(t_file)
		dist = calc_distance(q_vector, t_vector)
		distances = distances + [(filename, dist)]
	# Sort the distances
	distances = sorted(distances,key=lambda x: x[1])
	for candidate in distances:
		result_file.write("%s: " % candidate[0])
		result_file.write("%f\n" % candidate[1])
	result_file.close()
	
# calc distance between 2 vectors (Euclidian)	
def calc_distance(q_vec, s_vec):
	cumulative = 0

	for i in range(0, len(q_vec)):
		x = q_vec[i]
		y = s_vec[i]
		cumulative = cumulative + (x - y)**2
	return math.sqrt(cumulative)
	
	
	
def main():
	for filename in os.listdir(query_dir):
		f = open(query_dir + filename, 'r')
		print 'Calculating distance to testset for file: ' + filename 
		evaluate_distances(f)
		
if __name__ == "__main__":
	main()