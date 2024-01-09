def read_yaml(file_path):
    import yaml
    # Open the YAML file
    with open(file_path, 'r') as file:
        # Load the YAML content
        yaml_data = yaml.safe_load(file)
    return yaml_data