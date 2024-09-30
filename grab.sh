#!/bin/bash

# Clone the repository
git clone https://github.com/2OsZI4ISYd/stepcutis.git

# Navigate into the cloned repository directory
cd stepcutis || exit

# Make the setup.sh script executable
chmod +x setup.sh

# Run the setup.sh script
./setup.sh
