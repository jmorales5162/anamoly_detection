# final_project_aa2

git clone https://github.com/jmorales5162/final_project_aa2.git
cd final_project_aa2
pip install -r requirements.txt
kaggle datasets download -d sujaykapadnis/emotion-recognition-dataset
unzip emotion-recognition-dataset.zip; rm emotion-recognition-dataset.zip
python main.py