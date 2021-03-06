################################################
###  AWESOME LIBRARY FOR THE MMR ASSIGNMENT  ###
################################################

import os
import shutil
import math
import random
import cPickle
import sys
import subprocess
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

# In case of the new approach with one fixed query set and query path, here the path
fixed_query_path = scan_path + "FixedQuerySet\\"
fixed_test_path = scan_path + "FixedTestSet\\" 

result_folder = "Results\\"

# Default set settings
# Training set = first n picza's
n = 30
# Run in the background (hiding the commandwindow)?
run_in_background = True

#test_size = 20

# Evolutionary algorithm 
no_parents = 3
no_population = 10
no_children = 5
mm_size = 70
query_size = 20

# no_mutation is already calculated automatically
mutation_factor = 1.0/6.0
no_mutation = int(round( mutation_factor * mm_size ))
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
    execute("facecorrespondence " + facecor_path)

    cleanup_filter(facecor_path, ["sel4"])
  

# 1. Build mm #
# Builds a mm of the trainingset of facenumbers
def build_mm (test_name, training_set):
    # Assumption: precalc_facecor has been run before
    # Make mm directory and copy sel4 files
    mm_path = test_path + test_name + "\\" + mm_folder
    if not os.path.exists(mm_path):
        os.makedirs(mm_path)
     
    # the sel4 files (but not mirrsel4 files) still have to be copied.
    mm_path = test_path + test_name + "\\" + mm_folder
    copy_files_filter(facecor_path, mm_path, [str(face) + "_piczaNormalized_sel4" for face in training_set])
    
    # Use mmbuild.exe
    execute("mmbuild " + mm_path)

    
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
    execute("morphfit " + query_path + " 1 " + mm_path + " " + principal_components)
    
    # Move *_final.params to Query directory
    move_files_filter(".\\", query_path, ["final.params"])
    
    # Apply morphfit to the  test set
    execute("morphfit " + testset_path + " 1 " + mm_path + " " + principal_components)
    
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
        
# Morphfits the scans that are a fixed queryset and testset and copies these
# .params to their respective destination folders (and deletes them
def morphfit_scans_fixed (test_name, mm_size):

    # Path to query and test directories
    query_path = test_path + test_name + "\\" + query_folder
    testset_path = test_path + test_name + "\\" + test_folder
    
    # Path to the morphable model
    mm_path = test_path + test_name + "\\" + mm_folder + "data.bin"
    
    principal_components = str(mm_size - 1)
    
    # Apply morphfit to the  query set
    execute("morphfit " + fixed_query_path + " 1 " + mm_path + " " + principal_components)
    
    if not os.path.exists(query_path):
        os.makedirs(query_path)
    
    # Move *_final.params to Query directory
    move_files_filter(".\\", query_path, ["final.params"])
    
    # Apply morphfit to the  test set
    execute("morphfit " + fixed_test_path + " 1 " + mm_path + " " + principal_components)
    
    if not os.path.exists(testset_path):
        os.makedirs(testset_path)
    
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
    #not needed in case of fixed query and testset
    #generate_random_testingsets (test_name, query_amount, training_set)
    morphfit_scans_fixed (test_name, len(training_set))
    evaluate_results(test_name)

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
    # How many members of an old generation should go on to the next
    no_survivors = no_population - no_children
    
    iteration = 0
    #iteration = 1

    generation_path = test_name + "\\" + "generation" + str(iteration) + "\\"
    
    # Generation 0: make a current_population from random MM's
    current_population = range(0, no_population)
    #current_population = [4,5,7,8,9,11,12,13,14]

  
    for member in current_population:       
            cleanup_exe()
            # choose random scans for each member of generation 0
            full_test(generation_path + str(member) + "\\", random.sample(xrange(477, 588), mm_size), query_size)      


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

    #Keep track of 'how to call' the next child
    next_identifier = no_population
    #next_identifier = 15
    
    # Keep running until killed
    while(True):                      
        iteration += 1
        generation_path_prev = generation_path
        generation_path = test_name + "\\" + "generation" + str(iteration) + "\\"

        # Remove old directories that are no longer needed
        # For space efficiency
        remove_dead(generation_path_prev)
        
        # The children are created by choosing the parents
        # and mixing them together(with a little mutation)
        new_offspring, parents = create_offspring(sorted_population, generation_path_prev)
        # Determine the survivors from the previous round
        old_survivors = sorted_population[:no_survivors]

        # Build the new population
        current_population = [] 
        # Test the children (and give them new identifiers)
        for child in new_offspring:
            full_test(generation_path + str(next_identifier) + "\\", child, query_size)
            current_population += [next_identifier]
            next_identifier += 1

        # Don't test the old survivors, just copy them
        #  (space inefficient but nice for data analysis afterwards)
        for survivor in old_survivors:
            shutil.copytree(test_path + generation_path_prev + str(survivor[0]),
                            test_path + generation_path + str(survivor[0]))
            current_population += [survivor[0]]      
        
        # Get fitness of this generation's members
        population_fitness = retrieve_fitness(current_population, generation_path)
 
        sorted_population = zip(current_population, population_fitness)
      
        # Determine the final ranking of this generation, to be used by the next generation
        sorted_population.sort(key=lambda x: x[1], reverse = True)
     
        # Writing to log
        log_file = open( mmr_path + "log.txt", "a")
        log_file.write("Generation: " + str(iteration))
        log_file.write("\n")
        print_list(log_file,  "Parents: ", parents)
        log_file.write("\n")
        print_tuple_list(log_file, "Distance: ", sorted_population)
        log_file.write("\n")
        log_file.close()

        # Some much needed cleanup of the exe directory
        cleanup_exe()
            
    log_file = open( mmr_path + "log.txt", "a")
    log_file.write("Found best possible morphable model: \n")
    log_file.write("NR " + str(sorted_population[0][0]) +  " with a distance of " + str(sorted_population[0][1]))
    log_file.close()
    print "Found best possible morphable model: ",
    print test_name + " "+ str(sorted_population[0][0])

    
# Removes all dead directories to save space
def remove_dead(previous_generation_path):
    for i in os.listdir(test_path + previous_generation_path):
        q_path = test_path + previous_generation_path + i + "\\QuerySet\\"
        t_path = test_path + previous_generation_path + i + "\\TestSet\\"
        if os.path.exists(q_path):
            shutil.rmtree(q_path)
        if os.path.exists(t_path):
            shutil.rmtree(t_path)
        #shutil.rmtree(test_path + previous_generation_path + i + "\\MorphableModel")
        
# Create offspring #
# Create offspring given t
def create_offspring(population, test_name):

    #parents = population[0:no_parents] HARDCODED PARENT CHOICE
    
    # The complete population size
    n = len(population)
    # Create probability distribution depending on fitness for each
    # potential parent. Divide by squared fittest candidate
    prob_set = [int((((x[1]*x[1])/ (population[0][1]*population[0][1]) )) * 100) for x in population]
    
    choice_set = [] 
    # depending of the fitness for each participant in the population
    # create the probability it should be chosen from a set
    for i in range(0, n):
        choice_set += [i] * prob_set[i]
    
    # the set of parents
    parents = [] 
    # now choose all parents 
    for j in range(0, no_parents):
        p = random.choice(choice_set)
        parents += [population[p][0]]
        # Remove parent that is already used from choise
        choice_set = filter(lambda v: v != p , choice_set)
        
    offspring = []
    
    for i in range(0, no_children):
        offspring.append(create_child(parents, test_name))
    print offspring
    return offspring, parents

# Creates a new child, consisting of parental scans and a specific mutated
# amount
def create_child(parents, test_name):
    child_mm = []
    parent_scan_list = [] 
    # Find out for each parents what scans were used in mmbuild
    for i in parents:
        parent_scan_list.append(find_mm_scans(test_name + str(i)))
    # The range of the mutation
    parent_scan_list.append(range(477, 588))

    not_mutated = mm_size - no_mutation
    
    # Let the probability of p1 and p2 be the same
    # ceil in case the amount is a float and we do
    # not want the mutation to be more prominent   
    # Division of parent probabilities
    amount = int(math.ceil(not_mutated * 1/float(len(parents))))
    
    #Sequence from which the random generator chooses
    
    seq = []
    for i in range(0, len(parents)):
        seq += [i] * amount
        
    seq += [len(parents)] * no_mutation
 
    # Each element has a chance
    # of being from p1, p2 or part of the mutation
    for i in range(0, mm_size):
        r = random.choice(seq)
        add_elements(parent_scan_list[r], child_mm)
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
        # Retrieve the MAR from the pickle
        mar = float( cPickle.load(open(path + "mean_average_rank.p")) )
        ranks += [1 / mar]

    return ranks

# Write the new average MAR for all members
def write_average_mar(members, evo_path, evo_path_prev = ""):
    for member in members:
        path = test_path + evo_path + str(member) + "\\" + "eval\\"
        prev_path = test_path + evo_path_prev + str(member) + "\\" + "eval\\"

        # No old MAR measurements available
        if (evo_path_prev == "" or not os.path.exists(prev_path)):
            new_samples = 1
            new_average_mar = float( cPickle.load(open(path + "mean_average_rank.p")) )
        # Calculate from old measurements
        else:
            # read...
            old_average_mar = float( cPickle.load(open(prev_path + "average_mar.p")) )
            new_mar = float( cPickle.load(open(path + "mean_average_rank.p")) )
            old_samples = int( cPickle.load(open(prev_path + "mar_samples.p")) )
            # caluclate...
            new_samples = old_samples + 1
            new_average_mar = (old_average_mar * old_samples + new_mar) / new_samples
            print "Old survivor", str(member), "went from", str(old_average_mar), "average MAR to", str(new_average_mar), "with a MAR of", str(new_mar), "and now" + str(new_samples) + "samples"

        # Now write the results
        cPickle.dump(new_average_mar, open(path + "average_mar.p", "wb"))
        f = open(path + 'average_mar.txt', 'wb')
        f.write(str(new_average_mar))
        cPickle.dump(new_samples, open(path + "mar_samples.p", "wb"))
        f = open(path + 'mar_samples.txt', 'wb')
        f.write(str(new_samples))        
    
    
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
        w = weights[i]                  #OLD WEIGHT DISTRIBUTION (1.0 / (i+1))
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

# Execute a system command
def execute(command):
    print "Executing", command
    ret = subprocess.call(command, shell=run_in_background)

# Set up the fixed query and testset
def setup_fixed_testingsets():
    fixed_query_path = scan_path + "FixedQuerySet"
    fixed_test_path = scan_path + "FixedTestSet"
    
    if os.path.exists(fixed_query_path) or os.path.exists(fixed_test_path):
        print "[ERROR] one or two of the Fixed set folders already exist!"
        return
    
    # The fixed query/testset contain faces 488 - 607 (inclusive)
    faces = range(588, 608)
    types = ["a","b","c","d","e"]
    query_scans = []
    count = 0
    # Query an even amount of each scan type
    for face in faces:
        scan = str(face) + types[count % len(types)] 
        query_scans.append(scan)  
        count += 1
        #print "Scan ", scan
    
    # Fill query folder
    os.makedirs(fixed_query_path)
    move_files_filter(mirror_path, fixed_query_path, query_scans)

    # Fill test folder
    os.makedirs(fixed_test_path)
    move_files_filter(mirror_path, fixed_test_path, faces)

    # Remove from facecor folder
    cleanup_filter(facecor_path, range(477, 588))


#precalc_facecor();
#cleanup_test("test1")    
#full_test("sigma_test", range(477, 477 + n), 2)

cleanup_exe()
evolutionary_algorithm("EvoAlg_70")


# TIME (Tim PC) - full_test_fast:
# facecor + mm (mostly facecor) -> 1 min =~                                       2 sec * |trainingset|
# the rest (mostly morphfit) -> 1:32     =~ 3.07 sec * (|queryset| + |testset|) = 18.4 sec * |queryset|
