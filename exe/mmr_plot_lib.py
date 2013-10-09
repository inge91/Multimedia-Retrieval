# Library regarding data plotting #
import os
import matplotlib.pyplot as plt
import numpy
import mmr_lib
import cPickle


# plot precision-recall #
# plots all precision and recall files that could be found
# in a single graph
def plot_precision_recall():
	for element in os.listdir(mmr_lib.test_path):
		if os.path.isdir(mmr_lib.test_path + element):
			rp_path = mmr_lib.test_path + element + "\\" + "rp_eval\\"
			if not os.path.exists(rp_path):
				continue
			p = cPickle.load(open(rp_path+"precision.p", "rb"))
			r = cPickle.load(open(rp_path+"recall.p", "rb"))
			plt.plot(r,p, 'o')
			
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Recall & Precision of various MM')
	print "Saving to file"
	plt.savefig('out.png')

	
plot_precision_recall()
	
