from tersus import Cleaner

cleaner = Cleaner("example/messy_students.csv")

# 1. Profiling and suggestions
suggestions = cleaner.suggest()
print(suggestions)

# 2. Range validation checks
cleaner.check_range(
    "Marks",
    min_value=0,
    max_value=100
)
cleaner.check_range(
    "Attendance",
    min_value=0,
    max_value=100
)

# 3. Save snapshot before cleaning
cleaner.save("example/before_cleaning.csv")

# 4. Execute controlled quality cleaning
cleaner.clean(
    column_names=True,
    text=True,
    duplicates=True,
    types={"Age": "int"},
    missing="median",
    categories=True,
    dates=True,
    emails=True,
    phones=True,
    ranges=True,
    outliers="cap",
)
print(cleaner.data)

# 5. Validate post-cleaning state
validation = cleaner.validate()
print(validation)

# 6. Generate report
cleaner.report()

# 7. Save cleaned dataset
cleaner.save("example/after_cleaning.csv")

# Baseline comparison
real = Cleaner("example/students.csv")
real.profile()
real.suggest()