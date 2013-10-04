################################################
###  AWESOME LIBRARY FOR THE MMR ASSIGNMENT  ###
################################################

import os
import shutil
import math

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
        # Rank scans

    ## Example plug-in functions
        # Example distance functions
        # Example evaluation functions

### Constants
n = 30
mmr_path = "..\\"
exe_path = mmr_path + "exe\\"
scan_path = mmr_path + "scans\\"
posenorm_path = scan_path + "All Pose Normalized\\"
handnorm_path = scan_path + "PICZA Hand-Normalized\\"
landmark_path = scan_path + "PICZA Landmarks Namechange\\"
# This is where each new morphable model and it's results will be placed
test_path = mmr_path + "tests\\"
# Below are the folders within each test folder
facecor_folder = "FaceCorrespondence\\"
mm_folder = "MorphableModel\\"
query_folder = "QuerySet\\"
test_folder = "TestSet\\"
result_folder = "Results\\"

###


## Big functions

# 1. Build mm #
# Fast, mostly hardcoded function to build a morphable model from a list of face numbers.
# Uses the constants, makes a MM out of the first 30 faces.
def build_mm_fast (test_name):
    # TODO: remove 'first 30 faces' hardcoding
    training_set = range(477, 477 + n)
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

    # Make mm directory and copy sel4 files
    mm_path = test_path + test_name + "\\" + mm_folder
    if not os.path.exists(mm_path):
        os.makedirs(mm_path)
     
    copy_files_filter(facecor_path, mm_path, ["sel4"])

    # Use mmbuild.exe
    command = "mmbuild " + mm_path
    print "Executing", command
    os.system(command)

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

# 1-3. Full test




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
	name = q_file.name[-27:-24]

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
		dist = calc_distance(q_vector, t_vector)
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
def calc_distance(q_vec, s_vec):
	cumulative = 0
	for i in range(0, len(q_vec)):
		x = q_vec[i]
		y = s_vec[i]
		cumulative = cumulative + (x - y)**2
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
	
# Rank scans

## Example plug-in functions
# Example distance functions
# Example evaluation functions

#build_mm_fast("test1")
morphfit_scans("test1")
