# Tradear 📈
Scientific-Programming semester project  

## 1. Quick start

```bash
# 1 – clone & enter the project
git clone https://github.com/LeonLeonski/Tradear.git
cd Tradear

# 2 - install dependencies
python -m pip install requests
python -m pip install pandas
python -m pip install scipy
python -m pip install scikit-learn

# 2 – run the data-to-dashboard pipeline
python getData.py                   # ➜ bc_data.xml
python calculateData.py             # ➜ calculated_bitcoin_data.json
python getMultiplePredictions.py    # ➜ combined_predictions.json