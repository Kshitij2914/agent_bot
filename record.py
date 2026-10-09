# Course Recommendation Tool

courses = [
    {
        "name": "Python Basics",
        "subject": "Programming",
        "level": "Beginner",
        "duration": "4 weeks"
    },
    {
        "name": "Web Development",
        "subject": "Web Development",
        "level": "Beginner",
        "duration": "6 weeks"
    },
    {
        "name": "Data Science with Python",
        "subject": "Data Science",
        "level": "Intermediate",
        "duration": "8 weeks"
    },
    {
        "name": "Machine Learning",
        "subject": "Artificial Intelligence",
        "level": "Advanced",
        "duration": "10 weeks"
    },
    {
        "name": "Generative AI Basics",
        "subject": "Artificial Intelligence",
        "level": "Intermediate",
        "duration": "6 weeks"
    },
    {
        "name": "SQL Fundamentals",
        "subject": "Database",
        "level": "Beginner",
        "duration": "4 weeks"
    }
]


def recommend_courses(interest, level):
    recommendations = []

    for course in courses:
        if (
            course["subject"].lower() == interest.lower()
            and course["level"].lower() == level.lower()
        ):
            recommendations.append(course)

    return recommendations


def show_recommendations(recommendations):
    if len(recommendations) == 0:
        print("\nNo matching courses found.")
        print("Try a different interest or skill level.")
        return

    print("\nRecommended Courses:")
    print("--------------------")

    for course in recommendations:
        print("Course:", course["name"])
        print("Subject:", course["subject"])
        print("Level:", course["level"])
        print("Duration:", course["duration"])
        print()


def main():
    print("================================")
    print("   Course Recommendation Tool")
    print("================================")

    print("\nAvailable interests:")
    print("1. Programming")
    print("2. Web Development")
    print("3. Data Science")
    print("4. Artificial Intelligence")
    print("5. Database")

    interest = input("\nEnter your interest: ")

    print("\nSkill levels:")
    print("Beginner")
    print("Intermediate")
    print("Advanced")

    level = input("Enter your skill level: ")

    recommendations = recommend_courses(interest, level)

    show_recommendations(recommendations)


if __name__ == "__main__":
    main()