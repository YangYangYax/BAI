# BAI

## 0. Install PyTorch and other dependencies

```bash
pip install -r requirements.txt
```

## 1. Download datasets

Download datasets from [Baidu Cloud](https://pan.baidu.com/s/1FbcuvFbLjS60wxXeDKQFbA?pwd=yeah) and put them into the `data/` folder.

## 2. Run the model

Run the model with a `.yaml` configuration file like the following:

```bash
python run.py fit --config src/configs/yelp/yelp_full.yaml
```

