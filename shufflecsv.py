import pandas as pd
import os

curpath = os.path.join(os.getcwd(),'train_simplified')
destpath = os.path.join(os.getcwd(), 'shuffled')

df = pd.concat([pd.read_csv(os.path.join(curpath,f), nrows = 20000) for f in os.listdir(curpath)])
shuffled = df.sample(n=6800000)
for i in range(340):
    print(i)
    chosen = shuffled.iloc[i*20000:(i+1)*20000]
    chosen.to_csv(os.path.join(destpath, "train_data_{}.csv".format(i+1)))
    

validate = pd.concat([pd.read_csv(os.path.join(curpath, f), nrows = 100, skiprows=[i for i in range(1, 20000+1)]) for f in os.listdir(curpath)])
validate_shuffle = validate.sample(n=34000)
validate_shuffle.to_csv(os.path.join(destpath,"valid_data.csv"))
