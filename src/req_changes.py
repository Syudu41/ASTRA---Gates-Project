import yaml

# Load the environment.yml file
with open('src/environment.yml') as file:
    env_data = yaml.safe_load(file)

# Open the requirements.txt file for writing
with open('src/requirements.txt', 'w') as req_file:
    # Iterate over the dependencies
    for dep in env_data['dependencies']:
        if isinstance(dep, str):
            # Write conda package as is (optional: convert to pip package name if known)
            req_file.write(dep + '\n')
        elif isinstance(dep, dict) and 'pip' in dep:
            # Write pip packages as is
            for pip_dep in dep['pip']:
                req_file.write(pip_dep + '\n')
