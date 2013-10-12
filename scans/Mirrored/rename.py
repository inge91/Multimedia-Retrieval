import os

def main():
	for i in os.listdir("."):
		if ".ply" in i:
			name = i[:-4]
			print name
			os.rename(i, name + "_mirrored.ply") 
	
if __name__ == "__main__":
	main()
	