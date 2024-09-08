# Tesis_HAM10000

# Installation

# On Mac:
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install -r dev-requirements.txt
pip3 install -e .
export $(cat .env | xargs)

# On Windows:
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
pip install -e .
export $(cat .env | xargs)