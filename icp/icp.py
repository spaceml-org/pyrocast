import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from itertools import permutations, combinations
#import utils.dataprep as dp
import dataprep as dp
from scipy.stats import norm
import math




def getTestProbs(cube: np.ndarray, labels: np.ndarray, event_df: pd.DataFrame, cluster_var: str, cluster: str, numTrees: int = 100, envVar: np.ndarray = None) -> np.ndarray:
    """
    Function to calculate the "out-of sample" probability of PyroCb ocurrence for a given cluster by training an RF model on observations from all other clusters and using model to predict for cluster observations
    Args:
        cube: n x p .matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        cluster: name of the cluster for which predictions will be calculated
        numTrees: the number of trees to use for each RF model
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.

    Returns:
       a vector of pyroCb "out-of-sample" probability of occurrence for observations IN CLUSTER
    """
    #print("cluster: ", cluster)
    indx_val, =  np.where(event_df[cluster_var]==cluster)
    indx_train = np.array(list(set(np.arange(0, cube.shape[0])).difference(indx_val)))
    
    X_train = cube[indx_train,:]
    X_val = cube[indx_val,:]
    y_train = labels[indx_train]
    y_val = labels[indx_val]
    
    if envVar is not None:
        envVar_train = envVar[indx_train,]
        envVar_val = envVar[indx_val,]
        X_train = np.concatenate([X_train, envVar_train], axis=1)
        X_val = np.concatenate([X_val, envVar_val], axis=1)
    
    event_df_train = event_df.iloc[indx_train]
    event_df_val = event_df.iloc[indx_val]
    
    clf = RandomForestClassifier(n_estimators=numTrees, max_depth=10, class_weight="balanced_subsample", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_val)
    return y_pred[:,1]

def getTestProbsWrapper(cube: np.ndarray, labels: np.ndarray, event_df: pd.DataFrame, cluster_var: str, indxIncl: list, posts: np.ndarray, numTrees: int = 100, envVar: np.ndarray = None) -> np.ndarray:
    """
        A wrapper function that calculates pyroCb "out-of-sample" probability of ocurrence by calling
        getTestProbs for each cluster.
    Args:
        cube: n x p .matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        indxIncl: indices of features to be used. Could refer to raw variables or statistics
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        numTrees: the number of trees to use for each RF model
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.

    Returns:
        a vector of pyroCb "out-of-sample" probability of occurrence for ALL observations
    """
    if posts is not None: 
        indxIncl = np.array(dp.flatten([np.arange(posts[i], posts[i+1]).tolist() for i in indxIncl]))
    
    clusts = np.sort(np.unique(event_df[cluster_var]))
    
    if len(indxIncl)>0:
        pred =  cube[:,indxIncl]
    else:
        np.random.seed(1234556)
        smpl = np.random.choice(cube.shape[0], size=cube.shape[0], replace=False)
        pred = cube[smpl,:]
        
    
    y_preds = [getTestProbs(pred, labels, event_df, cluster_var, i, numTrees=numTrees, envVar=envVar) for i in clusts]
    y_pred = np.ones(cube.shape[0])*-1
    for i in clusts:
        indx_val, =  np.where(event_df[cluster_var]==i)
        y_pred[indx_val] = y_preds[i]
    return y_pred


def equalAUC_hypTest(y_pred_noE: np.ndarray, y_pred_E: np.ndarray, labels: np.ndarray) -> dict:
    """
    Performs conditional Independence test y indep E | X by comparing "reduced" model that excludes enviroment features E with "full" model that includes them
    Use deLong test for difference in AUC between classification models.
    Y is pyroCb ocurrence, X are the geostationary and
    meteorological variables and E are the environment variables (latitude, longitude and julian date)
    Args:
        y_pred_noE: probability of pyroCb ocurrence for each observation from "reduced" model excluding environment E features
        y_pred_E: probability of pyroCb ocurrence for each observation from "full" model including environment E features
        labels:  pyroCb ocurrence indicator for each observation

    Returns:
        dictionary with statistic for test, one-tail and two-tail p-value for test, aswell as the "full" and "reduced" AUCs
    """
    # without E
    y_pred1 = y_pred_noE[labels==1]
    y_pred0 = y_pred_noE[labels==0]
    Phi = (y_pred1[:,None] > y_pred0[None,:])
    VT_noE_1 = (1/(y_pred0.shape[0]-1))*np.apply_along_axis(np.sum, 1, Phi)
    VT_noE_0 = (1/(y_pred1.shape[0]-1))*np.apply_along_axis(np.sum, 0, Phi)
    A_noE = (np.sum(VT_noE_1)/VT_noE_1.shape[0]+np.sum(VT_noE_0)/VT_noE_0.shape[0])/2
    #print("auc without E:",A_noE)
    ST_noE_1 = (VT_noE_1-A_noE)**2
    ST_noE_1 = np.sum(ST_noE_1)/(ST_noE_1.shape[0]-1)
    ST_noE_0 = (VT_noE_0-A_noE)**2
    ST_noE_0 = np.sum(ST_noE_0)/(ST_noE_0.shape[0]-1)
    VA_noE = (ST_noE_1/VT_noE_1.shape[0])+(ST_noE_0/VT_noE_0.shape[0])
    # with E
    y_pred1 = y_pred_E[labels==1]
    y_pred0 = y_pred_E[labels==0]
    Phi = (y_pred1[:,None] > y_pred0[None,:])
    VT_E_1 = (1/(y_pred0.shape[0]-1))*np.apply_along_axis(np.sum, 1, Phi)
    VT_E_0 = (1/(y_pred1.shape[0]-1))*np.apply_along_axis(np.sum, 0, Phi)
    A_E = (np.sum(VT_E_1)/VT_E_1.shape[0]+np.sum(VT_E_0)/VT_E_0.shape[0])/2
    #print("auc with E: ",A_E)
    ST_E_1 = (VT_E_1-A_E)**2
    ST_E_1 = np.sum(ST_E_1)/(ST_E_1.shape[0]-1)
    ST_E_0 = (VT_E_0-A_E)**2
    ST_E_0 = np.sum(ST_E_0)/(ST_E_0.shape[0]-1)
    VA_E = (ST_E_1/VT_E_1.shape[0])+(ST_E_0/VT_E_0.shape[0])
    # Covariance
    ST_EnoE_1  = (VT_noE_1-A_noE)*(VT_E_1-A_E)
    ST_EnoE_1 = np.sum(ST_EnoE_1)/(ST_EnoE_1.shape[0]-1)
    ST_EnoE_0  = (VT_noE_0-A_noE)*(VT_E_0-A_E)
    ST_EnoE_0 = np.sum(ST_EnoE_0)/(ST_EnoE_0.shape[0]-1)
    COV_EnoE = (ST_EnoE_1/VT_E_1.shape[0])+(ST_EnoE_0/VT_E_0.shape[0])
    # Statistic
    V_noE_E = VA_noE + VA_E - 2*COV_EnoE
    z = (A_E-A_noE)/np.sqrt(V_noE_E)
    # H0: with E is not better -> A_E-A_noE is not large
    pval1tail = 1-norm.cdf(z)
    # H0: neither model is better
    pval2tail = (1-norm.cdf(np.abs(z)))+norm.cdf(-np.abs(z))
    res = {"stat":z, "pval_1tail":pval1tail, "pval_2tail":pval2tail,"auc_E":A_E, "auc_noE":A_noE}
    return res


def getHypWrapper(cube: np.ndarray, labels: np.ndarray, envVar: np.ndarray, event_df: pd.DataFrame, cluster_var: str, indxIncl: list, posts: np.ndarray, numTrees: int = 100) -> dict:
    """
    Wrapper that calculates probability of pyroCb occurrence for "full" (including E features) and "reduced" (excluding E features)
    and uses these to test conditional independence of Y and E given X.
    Y is pyroCb ocurrence, X are the geostationary and
    meteorological variables and E are the environment variables (latitude, longitude and julian date)
    Args:
        cube: n x p .matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        indxIncl: indices of features to be used. Could refer to raw variables or statistics
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        numTrees: the number of trees to use for each RF model


    Returns:
        dictionary with statistic for test, one-tail and two-tail p-value for test, aswell as the "full" and "reduced" AUCs
    """
    y_pred_noE = getTestProbsWrapper(cube, labels, event_df, cluster_var, indxIncl, posts, numTrees=numTrees)
    y_pred_E = getTestProbsWrapper(cube, labels, event_df, cluster_var, indxIncl, posts, numTrees=numTrees, envVar=envVar)
    res = equalAUC_hypTest(y_pred_noE, y_pred_E, labels)
    return res

def exclude_i(cube: np.ndarray, labels: np.ndarray, envVar: np.ndarray, event_df: pd.DataFrame, cluster_var: str, indxIncl: list, posts: np.ndarray, i: int, numTrees: int = 100) -> dict:
    """
    Performs hypothesis test for conditional independence Y indep E | X\i , i.e. excluding variable
    with index i from the current list of included variables indxIncl

    Args:
        cube: n x p matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        indxIncl: indices of features not excluded up to now. Could refer to raw variables or statistics
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        i: index of variable to be excluded from X
        numTrees: the number of trees to use for each RF model

    Returns:
       dictionary with statistic for test, one-tail and two-tail p-value for test, aswell as the "full" and "reduced" AUCs
    """
    print("i: ", i, " out of : ", len(indxIncl))
    indxExcl = indxIncl[i]
    indxIncl = list(set(indxIncl).difference(set([indxExcl])))
    res = getHypWrapper(cube, labels, envVar, event_df, cluster_var, indxIncl, posts, numTrees=numTrees)
    print(res)
    return res

def greedyICP(cube: np.ndarray, labels:np.ndarray, envVar: np.ndarray, event_df: pd.DataFrame, cluster_var: str, indxIncl: list, posts: np.ndarray, varss: list, numTrees: int = 100,
              pvalTest: str = "pval_1tail") -> dict:
    """
    This function implements greedy ICP algorithm by sequentially removing that variable i from the list
    of predictors have not been excluded yet, such that by excluding it we obtain the largest p-value
    for the conditional independence test Y indep E | X\i

    Args:
        cube: n x p matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        indxIncl: indices of features to be used. Could refer to raw variables or statistics
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        varss: list of strings denoting the geostationary and meteorological variables in the cube
        numTrees: the number of trees to use for each RF model
        pvalTest: string indicating if the pvalue should be 1-tailed ('pval_1tail') or 2-tailed ("pval_2tail')

    Returns:
        A dictionary with the sequences of excluded variable (as indices of the varss list),
        , corresponding p-values an aucs (for full and restricted RF models)
    """
    indxExcl = []
    indxPvals = []
    aucs_E = []
    aucs_noE = []
    indxIncl2 = indxIncl.copy()
    
    res = getHypWrapper(cube, labels, envVar, event_df, cluster_var, indxIncl, posts, numTrees=numTrees)
    print(res)

    while len(indxIncl2)>1:
        
        print("num left: ", len(indxIncl2))
        hyps = [exclude_i(cube, labels, envVar, event_df, cluster_var, indxIncl2, posts, i, numTrees=numTrees) for i in range(len(indxIncl2))]
        
    
        # index as a function of indices that are left. ie indxIncl2
        exclude_indx = np.argmax([h["pval_1tail"] for h in hyps])
        
        # index as a function of original indices
        exclude_indx2 = indxIncl2[exclude_indx]
        
        auc_E = hyps[exclude_indx]["auc_E"]
        auc_noE = hyps[exclude_indx]["auc_noE"]
        pval = np.max([h[pvalTest] for h in hyps])
        print(" exclude var: ", varss[exclude_indx2], " pval: ", pval, " auc_noE: ", auc_noE, " auc_E: ", auc_E)
        indxExcl.append(exclude_indx2)
        indxPvals.append(pval)
        aucs_E.append(auc_E)
        aucs_noE.append(auc_noE)
        #print("exclude_indx: ", exclude_indx)
        #print("indxIncl2 : ", indxIncl2)
        indxIncl2.pop(exclude_indx)
        
    indxExcl.append(indxIncl2[0])
    indxPvals.append(np.nan)
    indxIncl2.pop(0)
    aucs_E.append(0)
    aucs_noE.append(0)
    
    res =  {"indxVar":indxExcl, "pval":indxPvals, "auc_E":aucs_E, "auc_noE":aucs_noE}
    return res

def findIndxAux(varss: list, combos: list, i: int) -> int:
    """
    For a list of variables combos, and index i referring to said list, translate this index int
    terms of a different, larger, super-set, list varss
    Args:
        varss: list of strings denoting the geostationary and meteorological variables in the cube
        combos: list with selected list of variables, usually a product of using greedyICP with a certain cutoff
        i: index of variable within combos, for which we want index with respect to varss

    Returns:
        integer position index of combos[i] within varss list
    """
    indx, = np.where(np.array(varss)==combos[i])
    return indx[0]

def getHypWrapper2(cube: np.ndarray, labels: np.ndarray, envVar: np.ndarray, event_df: pd.DataFrame, cluster_var: str, varss: list, combos: list, j: int, posts: np.ndarray, numTrees: int = 100) -> dict:
    """
    Applies conditional independence test Y indep E | X where the features of X are taken from the
    j-th entry of a list of combinations of features of interest.
    Args:
        cube: n x p matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        varss: list of strings denoting the geostationary and meteorological variables in the cube
        combos: list of lists. Each inner list corresponds to a set of features to define X, and carry out Y indep E |X conditional independnece test
        j: the position index within combos for the set of features that defines X
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        numTrees: the number of trees to use for each RF model

    Returns:
        dictionary with statistic for test,  0ne-tail and two-tail p-value for test, aswell as the "full" and "reduced" AUCs
        Within the conditional independence test Y indep E|X, the features that define X are taken from
        the j-th entry of combos
    """
    print("j: ", j, " out of ", len(combos))
    print("combo: ",combos[j])
    indxIncl = [findIndxAux(varss, combos[j], i) for i in range(len(combos[j]))]
    res = getHypWrapper(cube, labels, envVar, event_df, cluster_var, indxIncl, posts, numTrees=numTrees)
    print("res: ")
    print(res)
    return res

def exhaustiveICP(cube: np.ndarray, labels: np.ndarray, envVar: np.ndarray, event_df: pd.DataFrame, cluster_var: str, varss: list, combos: list, posts: np.ndarray, numTrees: int) -> list:
    """
    Carries out exhaustive ICP. That is  it conditional independence tests of the form Y indep E|X_i
    are formed for i in 1,2,...,len(combos). The features of X_i are taken from the i-th entry of combos
    Args:
        cube (numpy array): n x p matrix with meteorological and geo-stationary featues for each site and time. These features are different statistics of each of 30 met. and geo-st. vars
        labels:  vector indicating whether a pyroCb occurred or not 6 hours after corresponding cube observation
        envVar: matrix with the features that define the E variable in ICP. If given will also be included in RF model.
        event_df: ancillary information for each observation including date, location, random and spatial clusters for CV, etc
        cluster_var: one of 'cluster_random' or 'cluster_regional' indicates whether CV is random or spatial
        varss: list of strings denoting the geostationary and meteorological variables in the cube
        combos: list of lists. Each inner list corresponds to a set of features to define X, and carry out Y indep E |X conditional independnece test
        posts: if indxIncl is given with respect to raw met. and geo. vars then posts include the indices where the statistics of each var begin and end
        numTrees: the number of trees to use for each RF model

    Returns:
        a list of dictionaries each of which contains results for each of the conditional independence test realized
    """
    res = [getHypWrapper2(cube, labels, envVar, event_df, cluster_var, varss, combos, j, posts, numTrees) for j in range(len(combos))]
    return res


def getSeqICP(varsSelec, varss,posts, cluster_var, cube, labels, event_df, num_reps, seed):
    print("varsSelec: ", varsSelec)
    indxIncl = np.sort([findIndxAux(varss, varsSelec, i) for i in range(len(varsSelec))]).tolist()
    indxIncl  = np.array(dp.flatten([np.arange(posts[i], posts[i+1]).tolist() for i in indxIncl]))
    res = [getFold(cluster, cluster_var, cube[:,indxIncl], labels, event_df, num_reps, seed, byinistate=False, importance=False) for cluster in np.unique(event_df[cluster_var])]
    res_msrs = [list(rm)[0] for rm in res]
    res_msrs = pd.concat(res_msrs)
    resMax = res_msrs[["rep","fold","auc","fpr","fnr"]].groupby(["fold"]).apply(maxAUC)#.reset_index()#.rename(columns={0:"auc"})
    res = {"mean":np.mean(resMax.auc), "std":np.std(resMax.auc)}
    print(res)
    return res




