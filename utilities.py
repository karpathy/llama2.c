import importlib
import pkg_resources

def check_dependencies():
    # Read the requirements from requirements.txt file
    with open('requirements.txt') as f:
        required_dependencies = f.read().splitlines()

    # Check if each dependency is installed
    missing_dependencies = []
    for dependency in required_dependencies:
        # Get the package name without version specification
        dependency_name = dependency.split('==')[0]
        if not pkg_resources.get_distribution(dependency_name).version:
            missing_dependencies.append(dependency)

    if missing_dependencies:
        # Print error message for missing dependencies
        print("Please download the following dependencies using pip:")
        for dependency in missing_dependencies:
            print(f"- {dependency}")
        # Exit the script with a non-zero exit code
        exit(1)

if __name__ == "__main__":
    check_dependencies()
