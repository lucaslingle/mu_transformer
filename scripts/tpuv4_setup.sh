set -e;
sudo apt install python3.10-venv -y;
python3 -m pip install --user pipx;
python3 -m pipx ensurepath;
"$HOME"/.local/bin/pipx install poetry;
