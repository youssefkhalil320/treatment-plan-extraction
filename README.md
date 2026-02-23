# Medical-Multi-turn

### create the environment 
```bash
conda create -n ann-ui python=3.10 -y
```

```bash
conda activate ann-ui
```

### install the requirements
```bash
pip install -r requirements.txt
```

## to change the hf cache
```bash
export HF_HOME="/mnt/data/users/youssef.mohamed/HF_CACHE"
```

```bash
salloc -N1 -n12 --mem=24G
source /apps/local/anaconda3/conda_init.sh
```

