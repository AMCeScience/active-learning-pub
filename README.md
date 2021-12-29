# Active learning miner

Developed and tested on Windows 10 inside a `venv` environment using Python 3.7.7 and pip 19.2.3

### Setup

Setup your environment by installing the requirements using pip.
```bash
pip install -r requirements.txt
```

1. Copy the [`config.example`](config.example) file to `config.py`.
```bash
cp config.example config.py
```

2. Run [`fetch_raw.py`](fetch_raw.py) to retrieve the data for the PubMed articles defined in the [qrel files](/Database/CLEF_data) from the [2017 CLEF eHealth Lab](https://github.com/CLEF-TAR/tar/tree/master/2017-TAR).
```bash
python fetch_raw.py
```

3. Run [`insert_docs.py`](insert_docs.py) to insert the fetched article data from PubMed into a [local database](/Database).
```bash
python insert_docs.py
```

4. Run [`fetch_validity.py`](fetch_validity.py) to check the database against the original qrel files.
```bash
python fetch_validity.py
```

5. Run [`clean_docs.py`](clean_docs.py) to preprocess the articles and store them as a feature matrix.
```bash
python clean_docs.py
```

### Experiments

1. Run [`run_experiments.py`](run_experiments.py) to determine the performance for baseline (using all data) and selected datasets.
```bash
python run_experiments.py
```

2. Run [`result_analysis.py`](result_analysis.py) to create plots and significance tests.
```bash
python result_analysis.py
```
