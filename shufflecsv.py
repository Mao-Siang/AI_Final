import pandas as pd
import os

curpath = os.path.join(os.getcwd(),'train_simplified')
destpath = os.path.join(os.getcwd(), 'shuffled')


for i in range(100):
    print(i)
    df = pd.concat([pd.read_csv(os.path.join(curpath,f), nrows = 500, skiprows=[j for j in range(1, 500*i+1)]) for f in os.listdir(curpath)])
    suffled = df.sample(n=100000)
    suffled.to_csv(os.path.join(destpath, "train_data_{}.csv".format(i+1)))
    

validate = pd.concat([pd.read_csv(os.path.join(curpath, f), nrows = 100, skiprows=[i for i in range(1, 50000+1)]) for f in os.listdir(curpath)])
validate_shuffle = validate.sample(n=34000)
validate_shuffle.to_csv(os.path.join(destpath,"valid_data.csv"))
