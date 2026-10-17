"""Complete end-to-end example demonstration of the Tersus data cleaning workflow.

Demonstrates the 7-step lifecycle:
load -> profile -> suggest -> clean -> validate -> report -> save
"""

from pathlib import Path
from tersus import Cleaner


def main():
    # File paths
    input_path = Path(__file__).parent / "students.csv"
    output_path = Path(__file__).parent / "output.csv"

    print("=" * 80)
    print("           TERSUS COMPLETE CLEANING LIFECYCLE DEMONSTRATION")
    print("=" * 80)

    # 1. LOAD: Initialize Cleaner with a CSV file or existing DataFrame
    print("\n[Step 1] Loading dataset...")
    cleaner = Cleaner(input_path)
    print(f"Loaded: {cleaner.data.shape[0]} rows, {cleaner.data.shape[1]} columns")

    # 2. PROFILE: Non-destructive quality inspection
    print("\n[Step 2] Profiling dataset quality characteristics...")
    profile = cleaner.profile()
    print(f"  - Missing values detected: {sum(profile.missing_counts.values())}")
    print(f"  - Duplicate rows detected: {profile.duplicate_rows}")
    print(f"  - Numeric columns: {', '.join(profile.numeric_columns)}")

    # 3. SUGGEST: Generate intelligent, prioritized cleaning recommendations
    print("\n[Step 3] Generating cleaning suggestions...")
    suggestions = cleaner.suggest()
    print(f"  Generated {len(suggestions)} suggestion(s):")
    for s in suggestions[:4]:
        col_str = f" in '{s.column}'" if s.column else ""
        print(f"  * [{s.severity.upper()}] {s.issue}{col_str}: {s.suggestion}")

    # 4. CLEAN: Execute orchestrated, controlled cleaning operations
    print("\n[Step 4] Executing controlled cleaning pipeline...")
    cleaner.clean(
        column_names=True,
        missing="median",
        duplicates=True,
        text=True,
    )
    print(f"  Completed {len(cleaner.history)} cleaning operation(s).")
    print(f"  Current dataset shape: {cleaner.data.shape[0]} rows x {cleaner.data.shape[1]} columns")

    # 5. VALIDATE: Post-cleaning goal satisfaction verification
    print("\n[Step 5] Validating post-cleaning dataset state...")
    validation = cleaner.validate()
    print(f"  - Goals satisfied: {validation.goals_satisfied}")
    print(f"  - Issues resolved: {len(validation.resolved_issues)}")
    print(f"  - Issues remaining: {len(validation.remaining_issues)}")
    print(f"  - Issues requiring manual review: {len(validation.manual_review_issues)}")

    # 6. REPORT: Comprehensive, beginner-friendly session summary
    print("\n[Step 6] Generating session report...")
    report = cleaner.report()
    print(report)

    # 7. SAVE: Export cleaned DataFrame to CSV
    print(f"\n[Step 7] Saving cleaned data to {output_path.name}...")
    cleaner.save(output_path)
    print("Saved successfully!")


if __name__ == "__main__":
    main()