source /cfarhomes/hzhou/.bashrc
conda activate RI_render
python script_landmark.py

conda deactivate 
source /cfarhomes/hzhou/.bashrc
conda activate RI_render
python script_generate_node.py
