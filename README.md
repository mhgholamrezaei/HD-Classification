This project is dedicated to implementation of classification algorithm using Hyperdimensional Computing (HDC) for various hardware platforms including CPU, GPU, and FPGA. 
## CPU 
### Step 0: Install The Packages
### Installed packages
1. **Python3** 
2. **Python sklearn**: `pip install scikit-learn`
3. **Python numpy**: 
4. **7zip**: `apt install p7zip-full p7zip-rar`


### Step1: Unpack The Dataset 

Install 7zip using this command:
```Bash
sudo apt install p7zip-full p7zip-rar
```
Extract the contents of the dataset:
```Bash 
cd dataset
7z x isolet_pickle.7z
```
Generate .pickle file:
```Python
Python dataConversion.py
```
### Step2: Run The Python Code:
```Bash 
python main.py --path ../dataset/isolet.pickle --d 1000 --alg rp --epoch 20
# Parameters: 
# --path: path to pickle dataset 
# --d: number of dimensions (defualt 500)
# --alg: encoding technique (rp, )
# --epoch: number of retraining iterations (default 20)
# --lr: learning rate (default 1.0)
# --L: number of levels (default 64)
```
Example output: 
```
Encoding 4991 train data
0% 4% 9% 14% 19% 24% 29% 34% 39% 44% 49% 54% 59% 64% 69% 74% 79% 84% 89% 94% 99%

Encoding 1247 validation data
0% 4% 9% 14% 19% 24% 29% 34% 39% 44% 49% 54% 59% 64% 69% 74% 79% 84% 89% 94% 99%

20 retraining epochs
epoch 0: 0.9054 epoch 1: 0.9118 epoch 2: 0.9102 epoch 3: 0.9166 epoch 4: 0.9198 epoch 5: 0.9206
epoch 6: 0.9182 epoch 7: 0.9166 epoch 8: 0.9198 epoch 9: 0.9214 epoch 10: 0.9182
epoch 11: 0.9222 epoch 12: 0.9206 epoch 13: 0.9254 epoch 14: 0.9222 epoch 15: 0.9230
epoch 16: 0.9278 epoch 17: 0.9246 epoch 18: 0.9246 epoch 19: 0.9254

Encoding 1559 test data
0% 4% 9% 14% 19% 24% 29% 34% 39% 44% 49% 54% 59% 64% 69% 74% 79% 83% 88% 93% 98%

0.9281590763309814
```
