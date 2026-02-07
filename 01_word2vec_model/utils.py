import os
import configparser


# Subclass ConfigParser to maintain case sensitivity
class MyConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr):
        return optionstr  # Maintain original case


# Function to update the pybliometrics.cfg file or create it if it doesn't exist
def update_pybliometrics_config(config_path, new_config):
    config = MyConfigParser()  # Use the custom config parser

    # Check if the config file exists
    if os.path.exists(config_path):
        # Read the existing config file
        config.read(config_path)
    else:
        # Create a new config file if it doesn't exist
        print(f"Config file not found at {config_path}. Creating a new one.")

    # Update sections based on the new_config dictionary
    for section, values in new_config.items():
        if not config.has_section(section):
            config.add_section(section)
        for key, value in values.items():
            if key == "APIKey" and isinstance(value, list):
                # Join list of API keys into a comma-separated string
                value = ",".join(value)
            config.set(section, key, str(value))

    # Write the updated (or new) config back to the file
    with open(config_path, "w") as configfile:
        config.write(configfile)
