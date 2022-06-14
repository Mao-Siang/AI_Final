import pandas as pd
import os

curpath = os.path.join(os.getcwd(),'train_simplified')
destpath = os.path.join(os.getcwd(), 'shuffled')

df = pd.concat([pd.read_csv(os.path.join(curpath,f), nrows = 4000) for f in os.listdir(curpath)])
shuffled = df.sample(n=4000*340)
for i in range(100):
    print(i)
    chosen = shuffled.iloc[i*13600:(i+1)*13600]
    chosen.to_csv(os.path.join(destpath, "train_data_{}.csv".format(i+1)))
    

validate = pd.concat([pd.read_csv(os.path.join(curpath, f), nrows = 100, skiprows=[i for i in range(1, 20000+1)]) for f in os.listdir(curpath)])
validate_shuffle = validate.sample(n=34000)
validate_shuffle.to_csv(os.path.join(destpath,"valid_data.csv"))
