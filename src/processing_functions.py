import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

import scipy as sp
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p

"""Here's a lot of functions that can be used in a processing pipeline
It aims to try different processing pipelines in ana easy fashion"""


# =============================================================================
# deal with nas
# =============================================================================


def fill_false_nas(df: pd.DataFrame):
    """a lot of nas are actually values representing the absence
    of a feature. We fill these false nas with correct values"""
    #avoid modifying the orginal dataframe
    tmp=df.copy()
    tmp["PoolQC"] = tmp["PoolQC"].fillna("No_Pool")
    
    tmp["MiscFeature"] = tmp["MiscFeature"].fillna("None")
    tmp["Alley"] = tmp["Alley"].fillna("None")
    tmp["Fence"] = tmp["Fence"].fillna("None")
    tmp["MasVnrType"] = tmp["MasVnrType"].fillna("None")
    tmp["MasVnrArea"] = tmp["MasVnrArea"].fillna(0)#0 surface when non masvnr
    
    tmp["FireplaceQu"] = tmp["FireplaceQu"].fillna("None")
    # categorical garage related vars have NA for no garage. 0 in numerical vars
    tmp["GarageType"] = tmp["GarageType"].fillna("None")
    tmp["GarageFinish"] = tmp["GarageFinish"].fillna("None")
    tmp["GarageQual"] = tmp["GarageQual"].fillna("None")
    tmp["GarageCond"] = tmp["GarageCond"].fillna("None")
    # for numerical values related to house without garage, some nas can be replaced by 0.
    tmp["GarageArea"] = tmp["GarageArea"].fillna(0)
    tmp["GarageYrBlt"] = tmp["GarageYrBlt"].fillna(0)
    tmp["GarageCars"] = tmp["GarageCars"].fillna(0)
    
    # basement
    tmp["BsmtQual"] = tmp["BsmtQual"].fillna("None")
    tmp["BsmtCond"] = tmp["BsmtCond"].fillna("None")
    tmp["BsmtExposure"] = tmp["BsmtExposure"].fillna("None")
    tmp["BsmtFinType1"] = tmp["BsmtFinType1"].fillna("None")
    tmp["BsmtFinType2"] = tmp["BsmtFinType2"].fillna("None")
    tmp["BsmtCond"] = tmp["BsmtCond"].fillna("None")
    
    # can assume nas are typical from documentation
    tmp["Functional"] = tmp["Functional"].fillna("Typ")
    
    # numerical columns with nas related to missing bsmt. fill with 0s
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        tmp[col] = tmp[col].fillna(0)
    # utils is not informative almost 100% same value  
    tmp=tmp.drop(['Utilities'],axis=1)
    return tmp

def fill_true_nas(df: pd.DataFrame):
    """some features have actual missing values that does not seem to be
    accurately and easily inferred from other features. We use various methods 
    to fill these nas"""
    
    tmp=df.copy() # avoid modifying the original df
    # we replace these nas with the most common value
    mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
    for col in mode_col:
        tmp[col] = tmp[col].fillna(tmp[col].mode()[0])
    #same filling with mode for this feature
    tmp["MSZoning"]=tmp["MSZoning"].fillna(tmp["MSZoning"].mode()[0])
    #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
    tmp["LotFrontage"] = tmp.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    return tmp
    

    
    # vis
    # var = 'has_pool'
    # data = pd.concat([tmp['PoolArea'], tmp[var]], axis=1)
    # data.plot.scatter(x=var, y='PoolArea', ylim=(0,1000));
def add_vars(df: pd.DataFrame):
    """We may add some features to make the model training
    better"""
    tmp=df.copy()#avoid modifying the source data frame
    # add a variable to specify if house has a pool
    tmp["has_pool"] = np.where(tmp['PoolQC'] == "No_Pool", 0, 1)
    
    ## Let's also add separate variable to spcify absense or presence of basement and garage.
    tmp["has_garage"] = np.where(tmp['GarageType'] == "None", 0, 1)
    tmp["has_bsmt"] = np.where(tmp['BsmtQual'] == "None", 0, 1)
    
    ## want total Surface
    tmp['TotalSF'] = tmp['TotalBsmtSF'] + tmp['1stFlrSF'] + tmp['2ndFlrSF']
    
    return tmp
## old handling only num inputs
# def normalize_min_max(df: pd.DataFrame):
#     """ Does a min max normalization on all columns of the dataframe
#     The input is a dataframe containing only numeric columns"""
#     tmp=df.copy()#copy to avoid modif original df
#     # Check if all columns in the DataFrame are numeric
#     if not all(pd.api.types.is_numeric_dtype(tmp[col]) for col in tmp.columns):
#         raise ValueError("All columns in the input DataFrame must be numeric")
    
#     tmp_norm=(tmp-tmp.min())/(tmp.max()-tmp.min())
#     return tmp_norm
def normalize_min_max(df: pd.DataFrame):
    """ Does a min max normalization on all columns of the dataframe
    The input is a dataframe containing only numeric columns"""
    tmp=df.copy()#copy to avoid modif original df
    # Check if all columns in the DataFrame are numeric
    # if not all(pd.api.types.is_numeric_dtype(tmp[col]) for col in tmp.columns):
    #     raise ValueError("All columns in the input DataFrame must be numeric")
    tmpnum=tmp.select_dtypes(include="number")
    tmpcatag=tmp.select_dtypes(exclude="number")
    tmpnum_norm=(tmpnum-tmpnum.min())/(tmpnum.max()-tmpnum.min())
    #concat num and categ df
    tmp_norm=pd.concat([tmpnum_norm, tmpcatag], axis=1)
    return tmp_norm
def normalize_robust(df: pd.DataFrame):
    """ applies robust scaler function from sklearn to 
    all numerical column in the dataframe
    
    requires:
    from sklearn.preprocessing import RobustScaler
    """
    tmp=df.copy()#copy to avoid modif original df
    # Check if all columns in the DataFrame are numeric
    # if not all(pd.api.types.is_numeric_dtype(tmp[col]) for col in tmp.columns):
    #     raise ValueError("All columns in the input DataFrame must be numeric")
    
    # separate num and categ vars
    tmpnum=tmp.select_dtypes(include="number")
    tmpcatag=tmp.select_dtypes(exclude="number")
    
    # instanciate scaler and fit
    scaler=RobustScaler().fit(tmpnum)
    # create scaled array
    scaled_array=scaler.transform(tmpnum)
    # convert back to dataframe
    tmpnum_norm = pd.DataFrame(scaled_array, columns=tmpnum.columns, index=tmpnum.index)

    #concat num and categ df
    tmp_norm=pd.concat([tmpnum_norm, tmpcatag], axis=1)
    return tmp_norm

def remove_outliers_picked(df: pd.DataFrame,y):
    """remove handpicked outliers
    NB: should only be applied to training and
    validation set. We want to make a prediction
    for all data point of test set, i.e. not remove rows
    
    input:
        df: the combined dataset with train and test
        (will handle train and test thanks to a col
         set_id in the df)
        y: the column of the dependent variable.
        need to include it as it is a criteria for outlier removal
    """
    tmp=df.copy()
    # take substet of data pertaining to training set
    #print(tmp.shape)
    # tmp["set_id"]=all_data["set_id"]
    tmp["SalePrice"]=y
    
    tmp_test=tmp[tmp['set_id'] == 'test']
    tmp=tmp[tmp['set_id'] == 'train']
    # print(tmp.shape)
    
    #remove outliers in GrLivarea
    tmp = tmp.drop(tmp[(tmp['GrLivArea']>4000) & (tmp['SalePrice']<300000)].index)
    # print(tmp.shape)
    #extract new dv column (only there on training set)
    y_new=tmp['SalePrice']
    
    # concatenate train and test
    tmp_all=pd.concat([tmp,tmp_test])
    # print(tmp_all.shape)
    
    # remove dv from feature df
    tmp_all = tmp_all.drop(columns=['SalePrice'])
    # print(tmp_all.shape)
    # raise SystemExit("This function is not implemented yet.")
    return tmp_all,y_new
    
    
def transform_distribution_features_simple(df: pd.DataFrame, lam=0.15):
    """do a box cox transformation on all numeric features
    with pre set parameter
    we just try to reduce slightly skewness of features
    
    required imports:
        from scipy.special import boxcox1p
        from scipy.stats import skew

    """
    
    tmp=df.copy()#copy to avoid modif original df

    # # Check if all columns in the DataFrame are numeric
    # if not all(pd.api.types.is_numeric_dtype(tmp[col]) for col in tmp.columns):
    #     raise ValueError("All columns in the input DataFrame must be numeric")
    
    # check which features are skewed
    #only look into numeric features
    numeric_feats = tmp.dtypes[tmp.dtypes != "object"].index
    skewed_feats = tmp[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    # print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})
   
    # select only most skewed features
    skewness_filt=skewness[abs(skewness["Skew"])>1]# abs to see positive and negative skew
   
    skewed_features = skewness_filt.index
    #apply box cox transform with preset lam parameter
    for feat in skewed_features:
        #all_data[feat] += 1
        tmp[feat] = boxcox1p(tmp[feat], lam)
    return tmp

def label_encoding_subset(df:pd.DataFrame):
    """we use label encoding for a subset of features
    
    requires:
        from sklearn.preprocessing import LabelEncoder
"""
    tmp=df.copy()#dont modify original data
    # print("shape befor processing:  ", tmp.shape)
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
            'YrSold', 'MoSold')
    # process columns, apply LabelEncoder to categorical features
    for c in cols:
        lbl = LabelEncoder() 
        lbl.fit(list(tmp[c].values)) 
        tmp[c] = lbl.transform(list(tmp[c].values))

    return tmp
def one_hot_encoding(df):
    tmp=df.copy()
    # remove set_id for processing
    set_id=tmp["set_id"]
    tmp=tmp.drop(columns=["set_id"])
    # one hot encode
    tmp = pd.get_dummies(tmp)*1#*1to turn booleans to num
    # tmp.shape
    # put set_id col back
    tmp["set_id"]=set_id
    return tmp
    
    
    
    
    
    