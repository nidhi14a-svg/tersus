from tersus import Cleaner

cleaner = Cleaner("example/students.csv")

profile = cleaner.profile()

print(profile)