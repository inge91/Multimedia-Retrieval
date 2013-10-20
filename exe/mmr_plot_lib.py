# Library regarding data plotting #
import os
import matplotlib.pyplot as plt
import numpy
import mmr_lib
import cPickle

# Hardcoded path to experiment dir (either absolute or relative)
test_path = "T:\\Documents\\UUstuff\\MMR\\"

path = "T:\\Documents\\UUstuff\\MMR\\Evo_90_fixed\\generation0\\"

# plot precision-recall #
# plots all precision and recall files that could be found
# in a single graph
def plot_precision_recall():
    for element in os.listdir(path):
        if os.path.isdir(path + element):
            #rp_path = mmr_lib.test_path + element + "\\" + "eval\\"
            rp_path = path + element + "\\" + "eval\\"
            if not os.path.exists(rp_path):
                continue
            p = cPickle.load(open(rp_path+"precision.p", "rb"))
            r = cPickle.load(open(rp_path+"recall.p", "rb"))
            print "Plotting " + element
            plt.plot(r,p, 'o-')
                       
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall & Precision of various MM')
    print "Saving to file"
    plt.savefig(path + '\\precision_recall.png')
    

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
def plot_evolution(plot_name, experiment_names, use_for_member = get_final_amar, use_for_generation = lambda l: min(l)):
    print "   ---   Plotting Evolution   ---   "
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
        plt.plot(plot_points, "o-")

    # Finally save the plot
    plt.xlabel('Generation')
    plt.ylabel('Generation value')
    plt.title('Progress of the evolutionary algorithm')
    print "Saving to file"
    plt.savefig(test_path + plot_name + '.png')

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
               use_for_generation = lambda l: min(l))
    plot_evolution("slide2", ["Evo_90_fixed"],
               use_for_member = get_current_amar,
               use_for_generation = lambda l: min(l))
    plot_evolution("slide3", ["Evo_90_fixed"],
               use_for_member = get_final_amar,
               use_for_generation = lambda l: min(l))
    plt.close()
    
    # Next compare the experiments with eachother (using final AMAR)
    # TODO add 30 and 60
    plt.figure()
    plot_evolution("slide4", ["Evo_90_fixed"],
           use_for_member = get_final_amar,
           use_for_generation = lambda l: min(l))
    plt.close()

    # Finally show precision/recall for the best of each experiment
    plt.figure()
    # TODO do precision/recall with upgraded method of Inge
    plt.close()
    


# Making some plots, in rough predicted order of smoothness -> chaos
# Current SIDE EFFECT of multiple plots: they will be added to eachother (but this can be useful for now)

plot_evolution("evo90_progress_finalAMAR_min", ["Evo_90_fixed"],
               use_for_member = get_final_amar,
               use_for_generation = lambda l: min(l))
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
make_slide_figures()




    
