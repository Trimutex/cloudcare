#!/bin/bash
echo "Updating the system..."
sudo apt-get update && sudo apt-get -y upgrade
echo "Installing packages..."
sudo apt-get install -y htop neovim wget
echo "Installing miniconda3..."
mkdir -p "$HOME/miniconda3"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
rm -rf "$HOME/miniconda3/miniconda.sh"
echo "Creating miniconda3 environment..."
"$HOME/miniconda3/bin/conda" init bash
source "$HOME/.bashrc"
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/bin/activate"
conda env create -f environment.yaml
conda activate cloudcare
echo "Setup complete!"
