if [ ! -d "sourcedata" ]; then
  mkdir sourcedata
fi

cd sourcedata

git clone https://github.com/google-research-datasets/dstc8-schema-guided-dialogue --depth 1

git clone https://github.com/budzianowski/multiwoz --depth 1

cd ..
export PYTHONPATH=.

python utils/sgd_data.py
python utils/multiwoz_data.py
