from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd  
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append('...')


epoch = 200
weak_label = 'compound' # MOA, treatment, none 


df_0 = pd.read_csv(f'NSCB_aggregated_features_weak_{weak_label}_0_DINO_epoch_{epoch}.csv')
label_df = df_0[["compound", "moa", "batch"]]
label_np = label_df.to_numpy()
feature_df0 = df_0.iloc[: , 1:-3]
df_1 = pd.read_csv(f'NSCB_aggregated_features_weak_{weak_label}_1_DINO_epoch_{epoch}.csv')
feature_df1 = df_1.iloc[: , 1:-3]
df_2 = pd.read_csv(f'NSCB_aggregated_features_weak_{weak_label}_2_DINO_epoch_{epoch}.csv')
feature_df2 = df_2.iloc[: , 1:-3]

feature_df = np.concatenate([feature_df0, feature_df1, feature_df2],axis=1)
feature_df = preprocessing.normalize(feature_df, norm='l2')
tosave_plot = feature_df
label_plot = label_df
tosave = np.concatenate([feature_df, label_np], axis = 1)
tosave = pd.DataFrame(tosave)
tosave.columns = [*tosave.columns[:-3],'compound', 'moa','batch']
tosave.to_csv(f"MEANaggregated_features_weak_{weak_label}_3ch_epoch_{epoch}.csv",index=True) #save to file




#######################
# NSC Matching function
#######################

feature_df = pd.DataFrame(feature_df)
print(feature_df)
print(label_df)
tally = []
for idx in range(len(label_df)):
    print(idx) # index
    feature = feature_df.iloc[[idx]] # feature vector
    same_compound_feat = label_df.iloc[[idx]]
    same_compound_val = same_compound_feat[["compound"]]
    same_compound_val = same_compound_val.to_numpy()
    same_compound_val = same_compound_val.item(0)
    print('feature')
    print(feature)
    drop_index = label_df.loc[label_df['compound'] == same_compound_val]
    drop_index = drop_index.index
    remaining_features1 = feature_df.drop(drop_index)
    remaining_features = remaining_features1#
    remaining_features = remaining_features.reset_index(drop=True)
 
    remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
    nn1 = remaining_features[['cos_sim']].idxmax()
    label_df_dropped = label_df.drop(drop_index)
    dif_compound_val = label_df_dropped.iloc[nn1]
    print('dif compound')
    print(dif_compound_val)
    moa_dif = dif_compound_val[["moa"]]
    moa_dif = moa_dif.to_numpy()
    moa_dif = moa_dif.item(0)

    moa_orig = same_compound_feat[["moa"]]
    moa_orig = moa_orig.to_numpy()
    moa_orig = moa_orig.item(0)
    print('moa_orig')
    print(moa_orig)
    print('moa_dif')
    print(moa_dif)
           
    if moa_orig == moa_dif:
        tally.append(1)
    else:
        tally.append(0)
    a_ret = np.mean(tally)
    print(a_ret)


#######################
# NSCB Matching function
#######################


feature_df = feature_df.drop([17,18,19,20,21,22,60,61,62,63,64], axis=0)#, inplace=True)
label_df = label_df.drop([17,18,19,20,21,22,60,61,62,63,64], axis=0)
label_np = label_df.to_numpy()


print(feature_df)
print(label_df)
tally = []
for idx in range(len(label_df)):
    print(idx) # index
    feature = feature_df.iloc[[idx]] # feature vector
    same_compound_feat = label_df.iloc[[idx]]
    same_compound_val = same_compound_feat[["compound"]]
    same_compound_val = same_compound_val.to_numpy()
    same_compound_val = same_compound_val.item(0)
    same_batch_val = same_compound_feat[["batch"]]
    same_batch_val = same_batch_val.to_numpy()
    same_batch_val = same_batch_val.item(0)
    drop_index1 = label_df.loc[label_df['compound'] == same_compound_val]
    remaining_features = feature_df.drop(drop_index1.index)
    label_df_dropped = label_df.drop(drop_index1.index)        
    drop_index2 = label_df_dropped.loc[label_df_dropped['batch'] == same_batch_val]
    label_df_dropped = label_df_dropped.drop(drop_index2.index)
    remaining_features1 = remaining_features.drop(drop_index2.index)
    remaining_features = remaining_features1     
    remaining_features['cos_sim'] = cosine_similarity(remaining_features, feature).reshape(-1)
    nn = remaining_features[['cos_sim']].idxmax()        
    dif_compound_val = label_df_dropped.loc[nn]
    print('dif compound')
    print(dif_compound_val)
    moa_dif = dif_compound_val[["moa"]]
    moa_dif = moa_dif.to_numpy()
    moa_dif = moa_dif.item(0)

    moa_orig = same_compound_feat[["moa"]]
    moa_orig = moa_orig.to_numpy()
    moa_orig = moa_orig.item(0)
    print('moa_orig')
    print(moa_orig)
    print('moa_dif')
    print(moa_dif)
           
    if moa_orig == moa_dif:
        tally.append(1)
    else:
        tally.append(0)
    a_ret = np.mean(tally)
    print(a_ret)



#######################
# Create TSNE Map
#######################


df = tosave_plot
y_hue = label_plot["moa"]
y_hue=y_hue.apply(str)
# 0 = Actin Disruptors
# 1 = Aurora kinase inhibitors
# 2 = Cholesterol-lowering
# 3 = DNA damage
# 4 = DNA replication
# 5 = Eg5 inhibitors
# 6 = Epithelial
# 7 = Kinase inhibitors
# 8 = Microtubule destabilizers
# 9 = Microtubule stabilizers
# 10 = Protein degradation
# 11 = Protein synthesis

print(y_hue)
x = df
print(x)
tsne = TSNE(n_components=2, verbose=1)#, random_state=123)
z = tsne.fit_transform(x) 

df2 = pd.DataFrame()
df2["y_hue"] = y_hue
df2["comp-1"] = z[:,0]
df2["comp-2"] = z[:,1]

sns_plot = sns.scatterplot(x="comp-1", y="comp-2", hue="y_hue",
                palette=sns.color_palette("hls", 12),
                data=df2, legend='full')

sns_plot.legend(fontsize='xx-small')

plt.savefig(f'tsne_{epoch}_{weak_label}.png')