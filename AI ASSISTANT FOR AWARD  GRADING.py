def get_grade(score):
    grade_scale = [
        ("A+", 90, 100),
        ("A", 75, 89),
        ("B", 60, 74),
        ("C", 40, 59),
        ("F", 0, 39)
    ]
    for letter, low, high in grade_scale:
        if low <= score <= high:
            return letter
    return "Invalid"


def generate_report():
    report = {}
    total_subjects = int(input("How many subjects? "))

    for idx in range(total_subjects):
        subj = input(f"Enter name of subject {idx+1}: ")
        score = int(input(f"Enter marks for {subj}: "))
        report[subj] = get_grade(score)

    print("\n===== REPORT CARD =====")
    for subj, grade in report.items():
        print(f"{subj} --> Grade: {grade}")


# Run program
generate_report()

