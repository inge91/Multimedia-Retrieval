#Simple name change script for landmarks
#That converts all names to the format [number]_piczaNormalized_lnd_extra.ply
import os
str = "_piczaNormalized_lnd_extra.ply"

def main():
	for filename in os.listdir("."):
		#disregard the script file itself
		if filename[-3:] == ".py":
			pass
		else:
			number = filename[7:10]
			os.rename(filename, number+str)
			
if __name__ == "__main__":
	main()