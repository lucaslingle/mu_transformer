set -e;
sudo apt update;
sudo apt install python3-venv -y;
sudo apt install python3.8-venv -y;
sudo apt install python3.9-venv -y;
python3 -m pip install --user pipx;
python3 -m pipx ensurepath;
"$HOME"/.local/bin/pipx install poetry;
