echo "5 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 5
echo "10 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 10
echo "15 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 15
echo "20 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 20
echo "25 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 25
echo "170 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 170
echo "200 dim"
python np_based/train.py --input_path notebooks/output/100-context-500000-data-38-questions/ --output_path notebooks/output/100-context-500000-data-38-questions/1/301k/ --n_embedding 200
