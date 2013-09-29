# Copies the faces ran through FaceCorrespondence with the right amount of verices
# to the Mmbuild directory
import os
import shutil

# the name of the files that should be copied end with this string
str = "_piczaNormalized_sel4.ply"

# Where to look for the files
src = "FaceCorrespondence\\"

# Destination for copied files
dst = "Mmbuild\\"

def main():
	for filename in os.listdir(src):
		if str in filename:
			shutil.copy(src + filename, "Mmbuild")

if __name__ == "__main__":
	main()