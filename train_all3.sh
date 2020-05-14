echo "30 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 30
echo "35 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 35
echo "40 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 40
echo "45 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 45
echo "75 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 75
echo "180 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 180
