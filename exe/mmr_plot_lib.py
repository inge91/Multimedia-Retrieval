# Library regarding data plotting #
import os
import matplotlib.pyplot as plt
import numpy
import mmr_lib
import cPickle

# Hardcoded path to experiment dir (either absolute or relative)
test_path = "F:\\MMR\\tests\\"

path = "F:\MMR\tests\evo_60_fixed\generation0\\"

# plot precision-recall #
# plots all precision and recall files that could be found
# in a single graph
def plot_precision_recall(plot_name, paths, file_names = []):
    artists = []
    artist_strings =[]
            #rp_path = mmr_lib.test_path + element + "\\" + "eval\\"
    for p in paths:
       
        rp_path = p + "\\eval\\"
 
        train_size = find_trainingset_size(p)
        
        if not os.path.exists(rp_path):
            continue
        precision = cPickle.load(open(rp_path+"precision.p", "rb"))
     
        recall = cPickle.load(open(rp_path+"recall.p", "rb"))

        pr = zip(precision,recall)
        
        pr_clean = diff_recall(pr)
        print pr_clean
        precision = [x[0] for x in pr_clean]
        recall = [x[1] for x in pr_clean]
        print "Plotting "
        f, = plt.plot(recall,precision, 'o-')
        artists += [f]
        artist_strings += ["MM with trainingsize " + str(train_size)]
        if len(file_names) > 0:
            cPickle.dump(recall, open(file_names[paths.index(p)] + "_recall.p", "w") ) 
            cPickle.dump(precision , open(file_names[paths.index(p)] + "_precision.p", "w") )
            # write legend entree
            f = open(file_names[paths.index(p)] + "_legend.txt", "w")
            f.write(artist_strings[paths.index(p)])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall & Precision of best MMS')
    plt.legend(artists, artist_strings)
    print "Saving to file"
    plt.savefig(plot_name + '.png')
    
    
def diff_recall(pr_list):
    recall = pr_list[0][1]
    final_pr = [(pr_list[0][0], pr_list[0][1])]
    for pr in pr_list:
        d_recall = recall - pr[1]
        if d_recall == 0:
            continue
        else:
            final_pr += [pr]
        recall = pr[1]
    print final_pr
    return final_pr

def find_trainingset_size(p):
    print p
    mm_path =  p + "\\" + "MorphableModel\\"
    mm_list = []
    # loop through all files
    for filename in os.listdir(mm_path):
        # if a file end with ply and begins with a number
        # add the number to the mm list
        if(filename.endswith(".ply")):
            if filename[0].isdigit():
                mm_list += [int(filename[0:3])]
    return len(mm_list)
     
    

# Look for the last generation where the member was present,
# to retrieve the final measurement of its Average MAR through the generations.
def get_final_amar(experiment_path, member, member_path):
    #print "Called with (", experiment_path, ",", member, ",", member_path, ")"
    gen = 0
    # Start searching from first generation, ignore given member_path
    member_path = experiment_path + "generation" + str(gen) + "\\" + member + "\\"
    
    # First, loop past the generation BEFORE the member was born
    while not os.path.isdir(member_path):
        gen += 1
        member_path = experiment_path + "generation" + str(gen) + "\\" + member + "\\"
        # Security check
        if gen >= 2000:
            print "[ERROR] Looking for member on path", member_path, "in generation 2000, member will probably not be found!"
            return 42.0
    
    # Then loop just past the final generation of the member
    while os.path.isdir(member_path):
        gen += 1
        member_path = experiment_path + "generation" + str(gen) + "\\" + member + "\\"
    # Then go back 1
    member_path = experiment_path + "generation" + str(gen-1) + "\\" + member + "\\"

    # Retrieve AMAR
    return float( cPickle.load(open(member_path + "eval\\average_mar.p", "rb")) )

# For each experiment (directory) given, plot the progress
# made by the evolutionary algorithm.
def plot_evolution(plot_name, experiment_names, use_for_member = get_final_amar, use_for_generation = lambda l: min(l), description = [], file_names = []):
    print "   ---   Plotting Evolution   ---   "
    legend = []
    for experiment_name in experiment_names:      
        # Try to find each experiment dir
        experiment_path = test_path + experiment_name + "\\"
        if not os.path.isdir(experiment_path):
            print "Directory", experiment_path, "not found!"
            continue
        print "EXPERIMENT:", experiment_name

        # Get one plot point out of each generation
        plot_points = []
        generation = 0
        while (os.path.isdir(experiment_path + "generation" + str(generation) + "\\")):
            print "Generation:", generation
            generation_path = experiment_path + "generation" + str(generation) + "\\"

            # Gather a value for each generation member
            member_values = []
            for member_folder in os.listdir(generation_path):
                member_path = generation_path + member_folder + "\\"
                if not os.path.isdir(member_path):
                    print "Skipping", member_path, ", this is not a directory!"
                    continue

                # Take the desired value of the member (default: final AMAR)
                member_value = use_for_member(experiment_path, member_folder, member_path)
                member_values.append(member_value)
                print "   added member", member_folder, "with value", member_value

            # Remember desired value for each generation (default: minimum (best) member value)
            generation_value = use_for_generation(member_values)
            print "---> generation", str(generation), "value = ", generation_value
            plot_points.append(generation_value)

            generation += 1
        
        # Plot the development of the min. AMAR over the generations
        legend += plt.plot(plot_points, "o-", label=description[experiment_names.index(experiment_name)])
        print len(plot_points)
        # if filenames are given, write the data to file
        if len(file_names) > 0:
            # write AMAR/MAR values
            cPickle.dump(plot_points, open(file_names[experiment_names.index(experiment_name)] + "_generation_value.p", "w") ) 
            cPickle.dump(range(0, len(plot_points)) , open(file_names[experiment_names.index(experiment_name)] + "_generations.p", "w") )
            # write legend entree
            f = open(file_names[experiment_names.index(experiment_name)] + "_legend.txt", "w")
            f.write(description[experiment_names.index(experiment_name)])
    plt.legend(loc='upper right', numpoints = 1)
    # Finally save the plot
    plt.xlabel('Generation')
    plt.ylabel('Generation value')
    plt.title('Progress of the evolutionary algorithm')
    print "Saving to file"
    plt.savefig(test_path + plot_name + '.png')
    
# plots evolution from files
# first argument x axis values
# second argument y axis value
# third argument legend
def plot_from_file(x_file, y_file, legend_file):
    print x_file
    f = open(x_file, 'r')
    x = cPickle.load(f)
    y = cPickle.load(open(y_file, 'r'))
    l = open(legend_file, 'rb')
    legend = l.readline()
    plt.plot(x, y, "o-", label=legend)
    
    
# Plugin functions for plot_evolution
def get_current_amar(experiment_path, member, member_path):
    return float( cPickle.load(open(member_path + "\\eval\\average_mar.p", "rb")) )

def get_mar(experiment_path, member, member_path):
    return float( cPickle.load(open(member_path + "\\eval\\mean_average_rank.p", "rb")) )
    
avg = lambda l: (sum(l)/float(len(l)))


# Find best member across generations in given experiment_path
#   best member has the LOWEST value for evaluate(member)
def find_best(experiment_path, evaluate = get_final_amar):
    best_score = 999999.9
    best_member = ""
    
    for generation_folder in os.listdir(experiment_path):
        generation_path = experiment_path + generation_folder + "\\"
        for member_folder in os.listdir(generation_path):
            member_path = generation_path + member_folder + "\\"
            if not os.path.isdir(member_path):
                    print "Skipping", member_path, ", this is not a directory!"
                    continue
            
            score = evaluate(experiment_path, member_folder, member_path)
            if score <= best_score:
                best_score = score
                best_member = member_folder
                print "Found better member", member_folder, "with score", score, "in generation", generation_folder

    # Return tuple of member and its score
    print "Final best member is", best_member, "with score", best_score
    return best_member, best_score;




def make_slide_figures():
    # Compare 'randomness' / steadiness (why was AMAR necessary)
    # Only for one of the experiments
    # TODO maybe use 30 for this example instead of 90
    plt.figure()
    plot_evolution("slide1", ["Evo_90_fixed"],
               use_for_member = get_mar,
               use_for_generation = lambda l: min(l), description = ["Minimal MAR"])
    plot_evolution("slide2", ["Evo_90_fixed"],
               use_for_member = get_current_amar,
               use_for_generation = lambda l: min(l), description = ["current AMAR"])
    plot_evolution("slide3", ["Evo_90_fixed"],
               use_for_member = get_final_amar,
               use_for_generation = lambda l: min(l), description = ["final AMAR"])
    plt.close()
    
    # Next compare the experiments with eachother (using final AMAR)
    # TODO add 30 and 60
    plt.figure()
    ######### THESE SHOULD BE UNCOMMENT WHEN YOU ADDED YOUR_PATH_HERE ###########
    #plot_from_file( "YOUR_PATH_HERE\\evo_60_fixed_final_amar_generations.p", "YOUR_PATH_HERE\\evo_60_fixed_final_amar_generation_value.p", "YOUR_PATH_HERE\\evo_60_fixed_final_amar_legend.txt")
    #plot_from_file( "YOUR_PATH_HERE\\evo_30_fixed_final_amar_generations.p", "YOUR_PATH_HERE\\evo_30_fixed_final_amar_generation_value.p", "YOUR_PATH_HERE\\evo_30_fixed_final_amar_legend.txt")
    plot_evolution("slide4", ["Evo_90_fixed"],
           use_for_member = get_final_amar,
           use_for_generation = lambda l: min(l), description = ["final AMAR for training size 90"])
    
    plt.close()

    # Finally show precision/recall for the best of each experiment
    plt.figure()
     ######### THESE SHOULD BE UNCOMMENT WHEN YOU ADDED YOUR_PATH_HERE ###########
    #plot_from_file( "YOUR_PATH_HERE\\evo_60_fixed_best_pc_recall.p", "YOUR_PATH_HERE\\evo_60_fixed_best_pc_precision.p", "YOUR_PATH_HERE\\evo_60_fixed_best_pc_legend.txt")
    #plot_from_file( "YOUR_PATH_HERE\\evo_30_fixed_best_pc_recall.p", "YOUR_PATH_HERE\\evo_30_fixed_best_pc_precision.p", "YOUR_PATH_HERE\\evo_30_fixed_best_pc_legend.txt")
    ######## ADD YOUR BEST PERFORMING ELEMENT HERE
    #plot_precision_recall("slide5", ["F:\\MMR\\tests\\evo_60_fixed\\generation40\\202"])
    
    # TODO do precision/recall with upgraded method of Inge
    plt.close()
    


# Making some plots, in rough predicted order of smoothness -> chaos
# Current SIDE EFFECT of multiple plots: they will be added to eachother (but this can be useful for now)
#
#plot_evolution("evo90_progress_finalAMAR_min", ["Evo_90_fixed"],
#               use_for_member = get_final_amar,
#               use_for_generation = lambda l: min(l))


'''
plot_evolution("evo90_progress_finalAMAR_avg", ["Evo_90_fixed"],
               use_for_member = get_final_amar,
               use_for_generation = avg)
plot_evolution("evo90_progress_currentAMAR_min", ["Evo_90_fixed"],
               use_for_member = get_current_amar,
               use_for_generation = lambda l: min(l))
plot_evolution("evo90_progress_currentAMAR_avg", ["Evo_90_fixed"],
               use_for_member = get_current_amar,
               use_for_generation = avg)
plot_evolution("evo90_progress_MAR_min", ["Evo_90_fixed"],
               use_for_member = get_mar,
               use_for_generation = lambda l: min(l))
plot_evolution("evo90_progress_MAR_avg", ["Evo_90_fixed"],
               use_for_member = get_mar,
               use_for_generation = avg)
'''

#find_best(test_path + "Evo_90_fixed\\")
#make_slide_figures()
#plot_evolution("slidebla", ["Evo_60_fixed"],
#           use_for_member = get_final_amar,
#           use_for_generation = lambda l: min(l), description = ["final amar"], file_names = ["evo_60_fixed_final_amar"])

#make_slide_figures()
#member, score = find_best("F:\\MMR\\tests\\evo_60_fixed\\")
#print member
plot_precision_recall(["F:\\MMR\\tests\\evo_60_fixed\\generation40\\202"], ["evo_60_fixed_best_pc"])