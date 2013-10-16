################################################
###  AWESOME LIBRARY FOR THE MMR ASSIGNMENT  ###
################################################

import os
import shutil
import math
import random
import cPickle
import sys
###  Contents
    ## Big functions

        # 1. Build mm

        # 2. Query the scans

        # 3. Evaluate query results

        # 1-3. Full test

    ## Utility functions
        # Copy files
        # Fit mm
        # Calculate distance
        # Evalution Functions

    ## Example plug-in functions
        # Example distance functions
        # Example evaluation functions

### Constants
mmr_path = "..\\"
exe_path = mmr_path + "exe\\"
scan_path = mmr_path + "scans\\"
posenorm_path = scan_path + "All Pose Normalized\\"
handnorm_path = scan_path + "PICZA Hand-Normalized\\"
landmark_path = scan_path + "PICZA Landmarks Namechange\\"
mirror_path = scan_path + "Mirrored\\"
facecor_path = scan_path + "Facecorresponded\\"
# This is where each new morphable model and it's results will be placed
test_path = mmr_path + "tests\\"
# Below are the folders within each test folder
facecor_folder = "FaceCorrespondence\\"
mm_folder = "MorphableModel\\"
query_folder = "QuerySet\\"
test_folder = "TestSet\\"
result_folder = "Results\\"

# Default set settings
# Training set = first n picza's
n = 30

#test_size = 20

# Evolutionary algorithm 
no_parents = 2
no_population = 10
no_mutation = 5
no_children = 5
mm_size = 41
max_iter = 9999
no_first_generation = 5
query_size = 5
###


## Big functions

# (0.)*  You only have to do this ONCE for the entire project #
# Calculates the facecorrespondence for all scans that could be in the trainingset,
#   this saves a lot of time, because facecorrespondence is one of the bottlenecks.
def precalc_facecor():
    facecor_path = scan_path + "Facecorresponded\\"
        
    # Check if the path exists
    if not os.path.exists(facecor_path):
        os.makedirs(facecor_path)

    # Copy the faces to be used in the MM
    copy_files_filter(handnorm_path, facecor_path, range(477,608))
    # Also copy the landmarks
    copy_files_filter(landmark_path, facecor_path, range(477,608))

    # Use facecorrespondence.exe
    command = "facecorrespondence " + facecor_path
    print "Executing", command
    os.system(command)

    cleanup_filter(facecor_path, ["sel4"])
  

# 1. Build mm #
# Builds a mm of the trainingset of facenumbers
def build_mm (test_name, training_set):
    '''
    facecor_path = test_path + test_name + "\\" + facecor_folder
    
    # Check if the path exists
    if not os.path.exists(facecor_path):
        os.makedirs(facecor_path)
    
    # Copy the faces to be used in the MM
    copy_files_filter(handnorm_path, facecor_path, training_set)
    # Also copy the landmarks
    copy_files_filter(landmark_path, facecor_path, training_set)

    # Use facecorrespondence.exe
    command = "facecorrespondence " + facecor_path
    print "Executing", command
    os.system(command)
    '''
    # Assumption: precalc_facecor has been run before
    # Make mm directory and copy sel4 files
    mm_path = test_path + test_name + "\\" + mm_folder
    if not os.path.exists(mm_path):
        os.makedirs(mm_path)
     
    # the sel4 files (but not mirrsel4 files) still have to be copied.
    mm_path = test_path + test_name + "\\" + mm_folder
    copy_files_filter(facecor_path, mm_path, [str(face) + "_piczaNormalized_sel4" for face in training_set])
    
    # Use mmbuild.exe
    command = "mmbuild " + mm_path
    print "Executing", command
    os.system(command)
    
# Fast, mostly hardcoded function to build a morphable model.
# Uses the constants, makes a MM out of the first n faces.
def build_mm_fast (test_name):
    build_mm(test_name, range(477, 477 + n))

# (*1.5 Generate random query- and testset*)
# Prepares a randomized queryset and testset in the right folders.
# The size is based on query_amount, with the testset size being based on the query set.
# Currently, none of the trainingset FACES is included in the queryset (is this good or bad?)
def generate_random_testingsets (test_name, query_amount, training_set):
	print "Generating random query and testset for test", test_name
	training_list_str = [str(num) for num in training_set]

    # Make a query- and matching testset,
    # queryset containing only Escans (as per example in the pdf),
    # and consisting of even amounts of all Escan versions
	query_scans = get_random_scans(query_amount, training_list_str, True, True)
	if len(query_scans) == 0:
		print "WARNING, QUERYSET EMPTY"
		print "test name: ",
		print test_name 
		print "query_amount: ",
		print quer_amount
		print "training_set ",
		print training_list_str
		sys.exit(1)
	test_scans = get_matching_testset(query_scans)   

    # Copy the sets to the right place
    # TODO: maybe try to make it work without copying, because the file sizes are not small
	query_path = test_path + test_name + "\\" + query_folder
	if not os.path.exists(query_path):
		os.makedirs(query_path)
	copy_files_filter(mirror_path, query_path, query_scans)
    
	testset_path = test_path + test_name + "\\" + test_folder
	if not os.path.exists(testset_path):
		os.makedirs(testset_path)
	# use mirrored data  
	copy_files_filter(mirror_path, testset_path, test_scans)     
    

# Fast version that assumes a MM consisting of the first n faces, and hardcoded size.
def generate_random_testingsets_fast (test_name):
    generate_random_testingsets (test_name, query_size, range(477, 477 + n))

# 2. Query the scans
# Applies morphfit using the existing data.bin file on both the query set and the test set
# to find the final params and writes these to the right directory
def morphfit_scans (test_name, mm_size):

    # Path to query and test directories
    query_path = test_path + test_name + "\\" + query_folder
    testset_path = test_path + test_name + "\\" + test_folder
    
    # Path to the morphable model
    mm_path = test_path + test_name + "\\" + mm_folder + "data.bin"
    
    principal_components = str(mm_size - 1)
    
    # Apply morphfit to the  query set
    command = "morphfit " + query_path + " 1 " + mm_path + " " + principal_components
    print "Executing", command
    os.system(command)
    
    # Move *_final.params to Query directory
    move_files_filter(".\\", query_path, ["final.params"])
    
    # Apply morphfit to the  test set
    command = "morphfit " + testset_path + " 1 " + mm_path + " " + principal_components
    print "Executing", command
    os.system(command)
    
    # Move *_final.params to Test directory
    move_files_filter(".\\", testset_path, ["final.params"])
    
    #Path to results. If it not yet exists, make it
    result_path = test_path + test_name + "\\" + result_folder
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    #loop through all files in query path
    for filename in os.listdir(query_path):
        # make sure the file is a .params
        if not ("final.params" in filename) :
            continue
        f = open(query_path + filename, 'r')
        print 'Calculating distance to test set for file: ' + filename 
        evaluate_distances(f, testset_path, result_path, test_name)

# 3. Evaluate query results
def evaluate_results(test_name):
	result_path = test_path + test_name + "\\" + result_folder
	mean_average_rank(result_path, test_name) 
	recall_precision_evaluation(result_path, test_name)
	mean_average_precision(result_path, test_name)
	last_rank_evaluation(result_path, test_name)

# 1-3. Full test
def full_test(test_name, training_set, query_amount):
    build_mm (test_name, training_set)
    generate_random_testingsets (test_name, query_amount, training_set)
    morphfit_scans (test_name, len(training_set))
    evaluate_results(test_name, )

# Uses the hardcoded functions
def full_test_fast (test_name):
    build_mm_fast (test_name)
    generate_random_testingsets_fast (test_name)
    morphfit_scans (test_name, n)
    evaluate_results(test_name)

# Evolutionary Algorithm #
# The evolutionary algorithm tries to find the best possible mask by 
# combining well performing parents to create new children masks
# keep top 30
# how many parents?
# how much mutation?
def evolutionary_algorithm(test_name):	
	iteration = 0
	generation_path = test_name + "\\" + "generation" + str(iteration) + "\\"
	# Create file for logging data
	
	# Generation 0: create n random morphable models
	for i in range(0, no_first_generation):
		# choose random numbers
		cleanup_exe()		
		full_test(generation_path + str(i) + "\\", random.sample(xrange(477, 608), mm_size), query_size)
		
	# create a list that contains current population
	i = no_first_generation
	
	current_population = range(0, i)
	#retrieves the fitness of current population 
	current_fitness = retrieve_fitness(current_population, generation_path)
	# sorts the population, best fitness in the front, and worst in the back
	sorted_population = zip(current_population, current_fitness)
	sorted_population.sort(key=lambda x: x[1], reverse = True)
	
	#Write data to log file
	log_file = open( mmr_path + "log.txt", "a")
	log_file.write("Generation: " + str(iteration))
	log_file.write("\n")
	print_tuple_list(log_file, "Distance: ", sorted_population)
	log_file.write("\n")
	log_file.close()
	
	cleanup_exe()
	
	# Stop condition: if not yet in the right error range or
	# The amount of iteratins has not yet ended
	# The optimal possible rank is 
	#while( if sorted_population[0][0] < 1/3.0 or iteration < 9999):
	while( iteration < 99):
		

		generation_path_prev = test_name + "\\" + "generation" + str(iteration) + "\\"
		iteration += 1
		generation_path = test_name + "\\" + "generation" + str(iteration) + "\\"
		
		# The children are created by choosing the parents
		# and mixing them together(with a little mutation)
		#TODO: Should this be prev?
		new_offspring, parents = create_offspring(sorted_population, generation_path_prev)
	
		children_population = [] 
		# Create the new morphable models of the children
		for child in new_offspring:
			full_test(generation_path + str(i) + "\\", child, query_size)
			children_population += [i]
			i += 1
		
		# Get fitness of children
		children_fitness = retrieve_fitness(children_population, generation_path)
		children_zipped = zip(children_population, children_fitness)
	
		# Before adding children evaluate previous generation again
		# to minimize randomization
		for l in sorted_population:
			scans = find_mm_scans(generation_path_prev + str(l[0]))
			full_test(generation_path + str(l[0]) + "\\", scans, query_size)
	
		# Add children to current population and remove all beings
		# above desired population size
		sorted_population += children_zipped
		
		sorted_population.sort(key=lambda x: x[1], reverse = True)
		sorted_population = sorted_population[0:no_population]

		# Writing to log
		log_file = open( mmr_path + "log.txt", "a")
		log_file.write("Generation: " + str(iteration))
		log_file.write("\n")
		print_list(log_file,  "Parents: ", parents)
		log_file.write("\n")
		print_tuple_list(log_file, "Distance: ", sorted_population)
		log_file.write("\n")
		log_file.close()
		# Remove all directories that no longer belong to the population
		# For space efficiency
		remove_dead(generation_path_prev, sorted_population)

		# Some much needed cleanup of the exe directory
		cleanup_exe()
	log_file = open( mmr_path + "log.txt", "a")
	log_file.write("Found best possible morphable model: \n")
	log_file.write("NR " + str(sorted_population[0][0]) +  " with a distance of " + str(sorted_population[0][1]))
	log_file.close()
	print "Found best possible morphable model: ",
	print test_name + " "+ str(sorted_population[0][0])


	
# Removes all dead directories to save space
def remove_dead(previous_generation_path, alive):
	for i in os.listdir(test_path + previous_generation_path):
		shutil.rmtree(test_path + previous_generation_path + i + "\\QuerySet\\")
		shutil.rmtree(test_path + previous_generation_path + i + "\\TestSet\\")
		#shutil.rmtree(test_path + previous_generation_path + i + "\\MorphableModel")
		
# Create offspring #
# Create offspring given t
def create_offspring(population, test_name):

	#parents = population[0:no_parents] HARDCODED PARENT CHOICE
	
	# The complete population size
	n = len(population)
	
	choice_set = []
	# depending of the fitness for each participant in the population
	# create the probability it should be chosen from a set we use 2**n for most likely
	# and 2 ** 1 for least likely
	for i in range(0, n):
		choice_set += [i] * (2**n-i)
	
	# the set of parents
	parents = [] 
	# now choose all parents 
	for j in range(0, no_parents):
		p = random.choice(choice_set)
		parents += [p]
		# Remove parent that is already used from choise
		choice_set = filter(lambda v: v != p , choice_set)
		
	offspring = []	
	# For all combination of parents create 
	# children
	previous = [] 
	for p1 in parents:
		for p2 in parents:
			if (p1 == p2) or (p2 in previous):
				continue
			# Create a specific number of children for each parent couple
			for i in range(0, no_children):
				offspring += [create_child(p1, p2, test_name)]
			previous += [p1] 
	return offspring, parents

# Creates a new child, consisting of parental scans and a specific mutated
# amound	
def create_child(p1, p2, test_name):
	child_mm = []
	# Find out for each parents what scans were used in mmbuild
	p1_l = find_mm_scans(test_name + str(p1))
	p2_l = find_mm_scans(test_name + str(p2))
	mutation = range(477, 608)
	
	not_mutated = mm_size - no_mutation  
	# Let the probability of p1 and p2 be the same
	# ceil in case the amount is a float and we do
	# not want the mutation to be more prominent
	p1_amount = int(math.ceil(not_mutated * 0.5))
	p2_amount = int(math.ceil(not_mutated * 0.5))
	
	#Sequence from which the random generator chooses
	seq = []
	seq += [0] * p1_amount
	seq += [1] * p2_amount
	seq += [2] * no_mutation
	
	# Each element has a chance
	# of being from p1, p2 or part of the mutation
	for i in range(0, mm_size):
		r = random.choice(seq)
		if r == 0:
			add_elements(p1_l, child_mm)
		if r == 1:
			add_elements(p2_l, child_mm)
		if r == 2:
			add_elements(mutation, child_mm)
	return child_mm
	
# Adds element from the random list to add_to_list
# without adding double values
def add_elements(random_list, add_to_list):
	selected_elements = [] 
	new_scan = random.choice(random_list)
	while(new_scan in add_to_list):
		new_scan = random.choice(random_list)
	selected_elements += [new_scan]
	add_to_list += [new_scan]
	return selected_elements
	
# find_mm_scans
# returns numbers of the scans with which the morphable model was made for given dir
def find_mm_scans(test_name):
	mm_path = test_path + test_name + "\\" + mm_folder
	mm_list = []
	# loop through all files
	for filename in os.listdir(mm_path):
		# if a file end with ply and begins with a number
		# add the number to the mm list
		if(filename.endswith(".ply")):
			if filename[0].isdigit():
				mm_list += [int(filename[0:3])]
	return mm_list
	
# Utility function
# retrieve fitness 
# returns the fitness for a list of candidates
def retrieve_fitness(candidates, evol_path):
	ranks = []
	for candidate in candidates:
		path = test_path + evol_path + str(candidate) + "\\" + "eval\\"
		# In case the path does not exist exit, as there is something wrong
		if not os.path.exists(path):
			print "ERROR: Fitness function could not be calculated as"
			print  path,
			print  "is missing.. Exiting"
			sys.exit(1)
		# return the rank from the pickle
		r = cPickle.load(open(path + "mean_average_rank.p"))
		ranks += [1 / float(r)]
	return ranks
	
## Utility functions

# Copy files #
# Copy files that have one of the strings in filter_list as a substring
def copy_files_filter (from_dir, to_dir, filter_list):
    print "Copying from", from_dir, "to", to_dir
    print "  with name filters:"
    print filter_list
    
    for filename in os.listdir(from_dir):
        copied = False;
        for name_filter in filter_list:
            if copied: break
            if str(name_filter) in filename and not copied:
                #print filename # <- optional spam print
                shutil.copy(from_dir + filename, to_dir)
                copied = True

# Remove every file from the directory,
# except files that have one of the strings in excluded_filters as a substring
def cleanup_filter(target_dir, excluded_filters):
    print "Cleaning up", target_dir
    print "  with excluded filters:"
    print excluded_filters
    
    for filename in os.listdir(target_dir):
        excluded = False;
        for name_filter in excluded_filters:
            if excluded: break
            if str(name_filter) in filename:
                excluded = True
        if not excluded:
            os.remove(target_dir + filename)

# Clean everything from the exe dir except for python and exe files
def cleanup_exe():
    cleanup_filter(exe_path, ["py", "exe"])

def cleanup_test(test_name):
	if os.path.exists(test_path + test_name):
		print "Removing test folder for test", test_name
		shutil.rmtree(test_path + test_name)
                
 # Move files #
 # Move files that have the substring filter_list in them from from_dir to to_dir
def move_files_filter (from_dir, to_dir, filter_list):
    print "Moving from", from_dir, "to", to_dir
    print "  with name filters:"
    print filter_list
    
    for filename in os.listdir(from_dir):
        for name_filter in filter_list:
            if str(name_filter) in filename:
                #print filename # <- optional spam print
                shutil.move(from_dir + filename, to_dir)
                
  # Evaluate distances #
 # Evaluates the distance between a single query files and all test files
def evaluate_distances(q_file, test_path, result_path, test_name):
	q_vector = file_to_list(q_file)

	basename = os.path.basename(q_file.name)
	name = basename[0:3]

    # create file containing ordered list of file rankings
	result_file = open(result_path + name + '.txt', 'w+')
    
    #list of distances
	distances = []
    
	for filename in os.listdir(test_path):
	# make sure the file is a .params
		if not ("final.params" in filename) :
			continue	
		t_file = open(test_path + filename, 'r')
		t_vector = file_to_list(t_file)
		dist = calc_euclidian_weighted_distance(q_vector, t_vector, test_name)
		distances = distances + [(filename, dist)]
    # Sort the distances
	distances = sorted(distances,key=lambda x: x[1])
	for candidate in distances:
		result_file.write("%s: " % candidate[0])
		result_file.write("%f\n" % candidate[1])
	# Last elements should be written without newline
	#result_file.write("%s: " % distances[-1][0])
    #result_file.write("%f" % distances[-1][1])
	result_file.close()
 
 
# Fit mm
# Calculate distance #
# Returns the Euclidian distance between 2 lists (interpreted as vectors)
def calc_euclidian_distance(q_vec, s_vec):
    cumulative = 0
    for i in range(0, len(q_vec)):
        x = q_vec[i]
        y = s_vec[i]
        cumulative = cumulative + (x - y)**2
    return math.sqrt(cumulative)
	
def calc_euclidian_weighted_distance(q_vec, s_vec, test_name):
	cumulative = 0
	# find weights
	weights = find_weights(test_name)
	for i in range(0, len(q_vec)):
		x = q_vec[i]
		y = s_vec[i]
		w = weights[i] 					#OLD WEIGHT DISTRIBUTION (1.0 / (i+1))
		cumulative = cumulative + w *(x - y)**2
	return math.sqrt(cumulative)	
	
def find_weights(test_name):
	# the file in which the weights are written
	mm_build = test_path + test_name + "\\" + mm_folder + "mmbuild.txt"
	f = open(mm_build, "r")
	f_complete = f.read()
	lines = f_complete.split("\n")
	# The third lines consists of the sigma values
	sigma_line = lines[2]
	#split the sigmas on tab values
	sigmas = sigma_line.split("\t")
	# make the values of the list consist of floats instead of string
	# exclude last one, as it is an empty string
	sigma_values = [float(i) for i in sigmas[:-1]]
	sigma_values += [0] * (99 - (n - 1))
	return sigma_values
	
# File to list #
# read in a file descriptor and return a list
def file_to_list(f):
    #loop through all
    values = f.readline()
    p = values.split('\t')
    # Remove the newline
    p = p[:-1]
    p = map(float, p)
    return p

# Return a list of size amount of randomized scan strings (e.g. ["477a", "506_picza"]).
# The scans strings in excluded_list are not chosen.
# A maximum of one scan per face will be picked (never "477a" and "477b").
# Setting evenly_distributed = True will return an equal amount of "a", "b", "_picza", etc. type scans.
#   WARNING: evenly_distributed is currently NOT guaranteed to work if specific scan strings are in excluded_set
#     (as opposed to only plain face numbers)
def get_random_scans (amount, excluded_list, escan_only = False, evenly_distributed = False):
    excluded_set = set(excluded_list)
    # Take a random sample from all face numbers MINUS the training_set
    # Strings are used, because e.g. scan "477b" could be excluded.
    face_numbers_str = [str(num) for num in range(477, 608)]
    potential_face_set = set(face_numbers_str) - excluded_set
    selected_faces = random.sample(potential_face_set, min([amount, len(potential_face_set)]))

    # Take one random scan per selected face
    scan_choices = ["a","b","c","d","e"]
    if not escan_only:
        scan_choices.append("_picza")
    if not evenly_distributed:
        scan_choices = set(scan_choices)
    
    selected_scans = []
    iteration = 0
    for face in selected_faces:
        # For an 'evenly distributed' set, iterate over the scan types instead of randomizing
        if evenly_distributed:
            chosen_scan = face + scan_choices[iteration % len(scan_choices)]
            if chosen_scan in excluded_set:
                print "[ERROR] Simple evenly_distributed implementation crashed, because chosen scan is in excluded list"
            else:
                selected_scans.append(chosen_scan)
            
        else:
            # Make sure that an excluded scan can not be picked
            potential_scans = set([(face + s) for s in scan_choices]) - excluded_set
            
            # It should never happen that all scans of a face have been used
            # (or else this code should be changed)
            if len(potential_scans) == 0: print "[ERROR] All scans for face", face, "have been excluded!"
            
            selected_scans.append(random.sample(potential_scans, 1)[0])
            
        iteration += 1

    return selected_scans

# Returns a testset consisting of all scans of the faces in queryset,
# but with actual scans that are in the query- or trainingset.
def get_matching_testset (queryset):
    testset = []
    for query_scan in queryset:
        if "_picza" in query_scan:
            query_face = query_scan[:-6]
        else:
            query_face = query_scan[:-1]
        for suffix in ["a","b","c","d","e", "_picza"]:
            testset.append(query_face + suffix)
            
    return list(set(testset) - set(queryset))

# Evaluation Functions #    
# lat_rank_evaluation evaluates all files in a directory using last_rank
def last_rank_evaluation(path, test_name):
	total_index = 0
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		index = last_rank(f, filename[0:3])
		total_index = total_index + index
	last_r = total_index / len(os.listdir(path)) 
	rp_path = test_path + test_name + "\\" + "eval\\"
	if not os.path.exists(rp_path):
		os.makedirs(rp_path)
	txt = open(rp_path + 'last_rank.txt', 'wb')
	cPickle.dump(last_r, open(rp_path + 'last_rank.p', 'wb'))
	txt.write(str(last_r))
	print "last rank =",
	print last_r
	
# last_rank takes the index of the last matching scan and its 
# distance and writes to file
def last_rank(file, number):
	ranks_str = file.read()
	ranks = ranks_str.split("\n")
	index = 0
	rank = 0
	for x in ranks:
		if number in x:
			rank = index + 1
		index = index + 1
	return rank

# for all files in results applies recall_precision and writes to evaluation
def recall_precision_evaluation(path, test_name):
	n = os.listdir(path)[0]
	f = open(path + n, 'r')
	ranks_str = f.read()
	ranks = ranks_str.split("\n")
	# Remove the last "" element
	ranks = ranks[:-1]
	l = len(ranks)
	
	# create total precision and recall list
	total_precision = [0] * l
	total_recall = [0] * l
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		p, r = recall_precision(f, filename[0:3])
		# add new calculated precision and recall to total
		total_precision = [i + j for i, j in zip(total_precision, p)]
		total_recall = [i + j for i, j in zip(total_recall, r)]
	
	# normalize values
	total_precision = [x/len(os.listdir(path)) for x in total_precision]
	total_recall = [x/len(os.listdir(path)) for x in total_recall]
	
	# check if there is already a recal precision dir
	rp_path = test_path + test_name + "\\" + "eval\\"
	if not os.path.exists(rp_path):
		os.makedirs(rp_path)
		
	# Pickle files for easy plotting access
	cPickle.dump(total_precision, open(rp_path + 'precision.p', 'wb'))
	cPickle.dump(total_recall, open(rp_path + 'recall.p', 'wb'))
	txt = open(rp_path + 'precision.txt', 'wb')
	print_list(txt, "", total_precision)
	
	txt2 = open(rp_path + 'recall.txt', 'wb')
	print_list(txt2, "", total_recall)
	
	print total_precision
	print total_recall
	
# At each index of the ranking calculate precision and recall
def recall_precision(file, number):
	ranks_str = file.read()
	ranks = ranks_str.split("\n")
	ranks = ranks[:-1]
	l = len(ranks)
	precisions = [0] * l
	recalls = [0] * l
	for i in range(0, l):
		r = relevant_shapes(ranks[0:i+1], number)
		precisions[i] = float(r) / (i + 1)
		m = relevant_shapes(ranks, number)
		recalls[i] = float(r) / m	
	return precisions, recalls

# Mean Average Precision #
# For each query result calculate the average precision and normalize 
# by query set size
# A value of 1 means a perfect performance by the mm (all matching masks are the first to be retrieved)
def mean_average_precision(path, test_name):
	q = len(os.listdir(path))
	total_average_precision = 0
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		total_average_precision += average_precision(f, filename)
	mean_average_precision = float(total_average_precision)/q
		# check if there is already a recal precision dir
	rp_path = test_path + test_name + "\\" + "eval\\"
	if not os.path.exists(rp_path):
		os.makedirs(rp_path)
		
	# Pickle files for easy plotting access
	cPickle.dump(mean_average_precision, open(rp_path + 'mean_average_precision.p', 'wb'))

	txt = open(rp_path + 'mean_average_precision.txt', 'wb')
	txt.write(str(mean_average_precision))

	
	print "Mean average precision: "
	print mean_average_precision
	
# Average Precision #
# calculate precision of query result
# at each position of the rank 
def average_precision(file, filename):
	average_precision = 0
	p, r = recall_precision(file, filename[0:3])
	for i in range(0, len(p)):
		if i > 0:
			diff_r = r[i] - r[i-1]
		else:
			diff_r = r[i]
		average_precision += p[i] * diff_r
	return average_precision
	
# returns the amount of elements in the list that
# match the term number 
def relevant_shapes(ranks, number):
	matches = 0 
	for i in ranks:
		if i.startswith(number):
			matches = matches + 1
	return matches

	# Calculates the mean average rank for all
	# query objects belonging to a single test
def mean_average_rank(path, test_name):
	q = len(os.listdir(path))
	total_average_rank = 0
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		total_average_rank += average_rank(f, filename)
	mean_average_rank = float(total_average_rank)/q
	mar_path = test_path + test_name + "\\" + "eval\\"
	if not os.path.exists(mar_path):
		os.makedirs(mar_path)
	cPickle.dump(mean_average_rank, open(mar_path + 'mean_average_rank.p', 'wb'))
	f = open(mar_path + 'mean_average_rank.txt', 'wb')
	f.write(str(mean_average_rank))
	print "Mean average rank: "
	print mean_average_rank

# Find the average of a rank in a single query retrieval
def average_rank(file, filename):
	ranks_str = file.read()
	ranks = ranks_str.split("\n")
	correct_ranks = []
	for i in range(0, len(ranks)):
		if ranks[i].startswith(filename[0:3]):
			correct_ranks += [i+1]
	return sum(correct_ranks)/float(len(correct_ranks))
	
# Write a list of tuples to a file descriptor
def print_tuple_list(f, text, tuple_list):
	f.write(text)
	f.write("[")
	for i in tuple_list:
		f.write("(")
		f.write(str(i[0]))
		f.write(" , ") 
		f.write(str(i[1]))
		f.write("), ")
	f.write("]")	
	
# Write a list to a file descriptor
def print_list(f, text, list):
	f.write(text)
	f.write("[")
	for i in list:
		f.write(str(i))		
		f.write(", ")
	f.write("]")		


#precalc_facecor();
#cleanup_exe()
#cleanup_test("test1")    
#full_test("sigma_test", range(477, 477 + n), 2)


evolutionary_algorithm("Evol7")


# TIME (Tim PC) - full_test_fast:
# facecor + mm (mostly facecor) -> 1 min =~                                       2 sec * |trainingset|
# the rest (mostly morphfit) -> 1:32     =~ 3.07 sec * (|queryset| + |testset|) = 18.4 sec * |queryset|
