################################################
###  AWESOME LIBRARY FOR THE MMR ASSIGNMENT  ###
################################################

import os
import shutil
import math
import random
import cPickle
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
query_size = 5
#test_size = 20

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
def morphfit_scans (test_name):

    # Path to query and test directories
    query_path = test_path + test_name + "\\" + query_folder
    testset_path = test_path + test_name + "\\" + test_folder
    
    # Path to the morphable model
    mm_path = test_path + test_name + "\\" + mm_folder + "data.bin"
    
    principal_components = str(n-1)
    
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
        evaluate_distances(f, testset_path, result_path)

# 3. Evaluate query results
def evaluate_results(test_name):
	result_path = test_path + test_name + "\\" + result_folder
	mean_average_rank(result_path, test_name) 

# 1-3. Full test
def full_test(test_name, training_set, query_amount):
    build_mm (test_name, training_set)
    generate_random_testingsets (test_name, query_amount, training_set)
    morphfit_scans (test_name)
    evaluate_results(test_name)

# Uses the hardcoded functions
def full_test_fast (test_name):
    build_mm_fast (test_name)
    generate_random_testingsets_fast (test_name)
    morphfit_scans (test_name)
    evaluate_results(test_name)

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
def evaluate_distances(q_file, test_path, result_path):
    q_vector = file_to_list(q_file)
    name = q_file.name[-28:-25]

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
        dist = calc_euclidian_weighted_distance(q_vector, t_vector)
        distances = distances + [(filename, dist)]
    # Sort the distances
    distances = sorted(distances,key=lambda x: x[1])
    for candidate in distances:
        result_file.write("%s: " % candidate[0])
        result_file.write("%f\n" % candidate[1])
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
	
def calc_euclidian_weighted_distance(q_vec, s_vec):
	cumulative = 0
	for i in range(0, len(q_vec)):
		x = q_vec[i]
		y = s_vec[i]
		w = (1.0 / (i+1))
		cumulative = cumulative + w *(x - y)**2
	return math.sqrt(cumulative)	
	

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
def last_rank_evaluation(path):
	total_index = 0
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		index = last_rank(f, filename[0:3])
		total_index = total_index + index
	last_r = total_index / len(os.listdir(path)) 
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
	rp_path = test_path + test_name + "\\" + "rp_eval\\"
	if not os.path.exists(rp_path):
		os.makedirs(rp_path)
		
	# Pickle files for easy plotting access
	cPickle.dump(total_precision, open(rp_path + 'precision.p', 'wb'))
	cPickle.dump(total_recall, open(rp_path + 'recall.p', 'wb'))
	print total_precision
	print total_recall
	
# At each index of the ranking calculate precision and recall
def recall_precision(file, number):
	ranks_str = file.read()
	ranks = ranks_str.split("\n")
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

def mean_average_rank(path, test_name):
	q = len(os.listdir(path))
	total_average_rank = 0
	for filename in os.listdir(path):
		f = open(path + filename, 'r')
		total_average_rank += average_rank(f, filename)
	mean_average_rank = float(total_average_rank)/q
	mar_path = test_path + test_name + "\\" + "mar_eval\\"
	if not os.path.exists(mar_path):
		os.makedirs(mar_path)
	cPickle.dump(mean_average_rank, open(mar_path + 'mean_average_rank.p', 'wb'))
	print "Mean average rank: "
	print mean_average_rank
	
def average_rank(file, filename):
	ranks_str = file.read()
	ranks = ranks_str.split("\n")
	correct_ranks = []
	for i in range(0, len(ranks)):
		if ranks[i].startswith(filename[0:3]):
			correct_ranks += [i+1]
	return sum(correct_ranks)/float(len(correct_ranks))
	



#generate_random_testingsets_fast("test1")
#morphfit_scans("test1")
cleanup_exe()
#evaluate_results("test1")
#cleanup_test("test1")    

full_test_fast("FTfast4")
'''
for size in range(5, 125, 5):
    full_test("First_" + str(size), range(477, size + 1), 10)
    '''  

#precalc_facecor();
#full_test("FullTest550-580_5", range(550, 581), 5)

# TIME (Tim PC) - full_test_fast:
# facecor + mm (mostly facecor) -> 1 min =~                                       2 sec * |trainingset|
# the rest (mostly morphfit) -> 1:32     =~ 3.07 sec * (|queryset| + |testset|) = 18.4 sec * |queryset|
