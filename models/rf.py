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



def getMsrs_rf(mod, X, y):

    y_pred = mod.predict_proba(X)
    
    weights = y*np.sum(y==0)+(1-y)*np.sum(y==1)
    
    #auc = roc_auc_score(y, y_pred[:,1], sample_weight=weights)
    auc = roc_auc_score(y, y_pred[:,1])
    
    thrs = 0.3#thresholds[np.argmax(f1_scores)]
    
    
    tn, fp, fn, tp = confusion_matrix(y, (y_pred[:,1]>thrs)*1).ravel()
    fpr = fp / (fp+tp)
    fnr = fn / (fn+tn)
    res = {"auc":auc, "fpr":fpr, "fnr":fnr}
    #print("fp: ", fp)
    #print("tp: ", tp)
    
    fpr, tpr,_ = roc_curve(y, y_pred[:,1])
    rocurve = {"fpr":fpr, "tpr":tpr}
    
    #print("res: ", res)
    return res, rocurve

def getIniStates(event_df):
    tab = event_df[["red_flag_now","flag_then","event_id"]].groupby(["red_flag_now","flag_then"]).count()
    tab = tab.reset_index()
    ini_states, cnts = np.unique(tab.red_flag_now, return_counts=True)
    tab = pd.DataFrame({"ini_state":ini_states, "cnt":cnts})
    valid_ini_states = tab.loc[tab.cnt==2].ini_state.tolist()
    return ini_states, valid_ini_states

def getMsrLoader_inistate(inistate, valid_ini_states, event_df, X, y, mod):
    if inistate not in valid_ini_states:
        return {"auc":np.nan, "fpr":np.nan, "fnr":np.nan}
    indx, = np.where(event_df.red_flag_now==inistate)
    
    #indx1, = np.where( (event_df.red_flag_now==inistate) & (event_df.flag_then))
    #indx0, = np.where( (event_df.red_flag_now==inistate) & (np.logical_not(event_df.flag_then)))
    num = X.shape[0]
    #num1 = np.sum(event_df.flag_then)
    #num0 = event_df.shape[0]-num1
    np.random.seed(seed=1234567)
    #indx = np.random.choice(indx, size=num, replace=True)
    #indx1 = np.random.choice(indx1, size=num1, replace=True)
    #indx0 = np.random.choice(indx0, size=num0, replace=True)
    #indx = np.concatenate([indx0,indx1], axis=0)
    
    def jitter(x):
        return x + np.random.normal(size=x.shape[0])*np.std(x)*0.05
    
    #X2 = np.apply_along_axis(jitter,0,X)
    
    msrs_inistate = getMsrs_rf(mod, X[indx,], y[indx,])
    res = list(msrs_inistate)[0]
    return res

def getMods(X_train, y_train,posts, i, C):
    #print("i : ", i, " out of: ", posts.shape[0]-1)
    #print("posts i -postis i+1: ", posts[i:(i+2)])
    indx_excl = np.arange(posts[i],posts[i+1])
    
    indx, = np.where(np.apply_along_axis(np.any,1,(C.to_numpy()[:,np.array(indx_excl)]>0.4)))
    indx_excl = np.sort(np.array(list(set(indx).union(indx_excl))))
    
    indx_incl = np.arange(posts[posts.shape[0]-1])
    indx_incl = np.array(list(set(indx_incl).difference(indx_excl)))
    
    
    mod_without = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight="balanced_subsample", random_state=0)
    mod_without.fit(X_train[:,indx_incl], y_train)
    
    return mod_without

def getModsWrapper(X_train, y_train, posts, C):
    mod_with = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight="balanced_subsample", random_state=0)
    mod_with.fit(X_train, y_train)
    mods_without = [getMods(X_train, y_train, posts, i, C) for i in range(posts.shape[0]-1)]
    res = {"mod_with": mod_with, "mods_without":mods_without}
    return res

def getRep(seed, X_tr, y_tr, X_te, y_te, event_df_te, rep, C=None, byinistate=True, importance=False, shuffle=False):
    
    print("*********************************")
    print("REP: ", rep)
    print("seed: ", seed)
    print("*********************************")
    
    if shuffle:
        smpl = np.random.choice(X_tr.shape[0], size=X_tr.shape[0], replace=False)
        X_tr2 = X_tr[smpl,:]
    else:
        X_tr2 = X_tr
    
    mod2 = RandomForestClassifier(n_estimators=500, max_depth=10, class_weight="balanced_subsample", random_state=seed)
    mod2 = mod2.fit(X_tr2, y_tr)
    
    print("num pyrocb train: ", np.sum(y_tr))
    print("num pyrocb test: ", np.sum(y_te))
    
    res, rocurve = getMsrs_rf(mod2, X_te, y_te)
    res["rep"] = rep
    
    
    
    
    
    
    out = [res, rocurve]
    
    if byinistate:
        ini_states, valid_ini_states = getIniStates(event_df_te)
                    
        res_inistate = [getMsrLoader_inistate(inistate, valid_ini_states, event_df_te, X_te, y_te, mod2) for inistate in ini_states]
        res_inistate = pd.DataFrame(res_inistate)
        res_inistate["rep"] = rep
        res_inistate["inistate"] = valid_ini_states
        out = out + [res_inistate]
    
    if importance:
        posts =np.arange(0,X_tr.shape[1]+1-10, 11)
        posts = np.array(posts.tolist()+[posts[posts.shape[0]-1]+4,posts[posts.shape[0]-1]+10])

        #imps = getModsWrapper(X_tr, y_tr, posts, C)
        
        #imps = permutation_importance(mod2, X_te, y_te, n_repeats=10, random_state=42, n_jobs=1)
        imps = mod2
        out = out + [imps]
    
    
    return tuple(out)


def getFold(cluster, cluster_var, cube, labels, event_df, num_reps, seed, byinistate=True, importance=False, shuffle=False):
    
    print("*********************************")
    print("FOLD: ", cluster)
    print("*********************************")
    
    df = pd.DataFrame(cube)
    C = df.corr(method="spearman")
    
    indx_val, =  np.where(event_df[cluster_var]==cluster)
    indx_train = np.array(list(set(np.arange(0, cube.shape[0])).difference(indx_val)))
    
    X_train = cube[indx_train,:]
    X_val = cube[indx_val,:]
    y_train = labels[indx_train]
    y_val = labels[indx_val]
    
    
    event_df_train = event_df.iloc[indx_train]
    event_df_val = event_df.iloc[indx_val]
    
    
    np.random.seed(seed=seed)
    seedInis = np.random.choice(100000, size=5).astype(int)
    
    
    
                
    res_reps = [getRep(seedInis[rep], X_train, y_train, X_val, y_val, event_df_val, rep, C, byinistate=byinistate, importance=importance, shuffle=shuffle) for rep in range(num_reps)]
    
    res_msrs = [rr[0] for rr in res_reps]
    rocurves = [rr[1] for rr in res_reps]
    
    res = pd.DataFrame(res_msrs)
    res["fold"] = cluster
    
    out = [res, rocurves]
    
    if byinistate:
        res_msrs_inistate = [rr[2] for rr in res_reps]
        res_inistate = pd.concat(res_msrs_inistate)
        res_inistate["fold"] = cluster
        out = out + [res_inistate]
    
    if importance:
        imps = [rr[len(rr)-1] for rr in res_reps]
        out = out + [imps]
    
    
    return tuple(out)


def getMarginalImpCluster(mods, X_val, y_val, posts, i, C):

    #print("i : ", i, " out of: ", posts.shape[0]-1)
    indx_excl = np.arange(posts[i],posts[i+1])
    indx, = np.where(np.apply_along_axis(np.any,1,(C.to_numpy()[:,np.array(indx_excl)]>0.4)))
    indx_excl = np.sort(np.array(list(set(indx).union(indx_excl))))
    
    indx_incl = np.arange(posts[posts.shape[0]-1])
    indx_incl = np.array(list(set(indx_incl).difference(indx_excl)))

    mod_with = mods["mod_with"]
    mod_without = mods["mods_without"][i]
    y_pred = mod_with.predict_proba(X_val)
    auc_with = roc_auc_score(y_val, y_pred[:,1])

    y_pred = mod_without.predict_proba(X_val[:,indx_incl])
    auc_without = roc_auc_score(y_val, y_pred[:,1])

    imp = auc_with-auc_without
    #print("importance: ", imp)
    return imp

def getMarginalImp(mods, cube, labels, cluster, event_df, posts, C):
    indx_val, =  np.where(event_df.cluster==cluster)
    indx_train = np.array(list(set(np.arange(0, cube.shape[0])).difference(indx_val)))
    
    X_train = cube[indx_train,:]
    X_val = cube[indx_val,:]
    y_train = labels[indx_train]
    y_val = labels[indx_val]

    imps = [getMarginalImpCluster(mods[cluster][3][0], X_val, y_val, posts, j, C) for j in range(posts.shape[0]-1)]
    return imps

def getTreeProbs(leaf_indx_tr, y_tr, ws, leaf_indx_pr):
    res = pd.crosstab(leaf_indx_tr, y_tr, ws[y_tr], aggfunc=sum, normalize="index")
    res = res.reindex(index=np.unique(leaf_indx_pr))  # leaves that do not have any element form training have a NAN
    res = res.reindex(columns=np.arange(2))
    # probability
    # res = res.reset_index()
    return res

def predProbRF(forestProbs_tr, Leaves_pred):
    # here we use the Leaves index Leaves_pred[:,i] to find the probability of the leaf

    def debugThis(forestProbs_tr, Leaves_pred, i):
        res = forestProbs_tr[i]
        res = res.loc[Leaves_pred[:, i], 1]
        res = np.array(res)

        return res

    probPred_forest = [debugThis(forestProbs_tr, Leaves_pred, i) for i in range(Leaves_pred.shape[1])]
    probPred_forest = np.array(probPred_forest).T

    probPred = np.apply_along_axis(np.nanmean, 1, probPred_forest)
    return probPred


def applyRF(clf, Xtr, Xpr, ytr, boots):
    Leaves_tr = clf.apply(Xtr)
    Leaves_pr = clf.apply(Xpr)
    n_classes = 2
    n_samples = Xtr.shape[0]
    ws = n_samples / (n_classes * np.bincount(ytr))

    def DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i):
        return getTreeProbs(Leaves_tr[boots[:, i], i], ytr[boots[:, i]], ws, Leaves_pr[:, i])

    forestProbs_tr = [DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i) for i in range(Leaves_tr.shape[1])]

    probPred = predProbRF(forestProbs_tr, Leaves_pr)
    return probPred

def applyRF_comb(clf1, clf2, Xtr, Xpr, ytr, indx1, indx2, boots):
    Leaves_tr1 = clf1.apply(Xtr[:,indx1])
    Leaves_pr1 = clf1.apply(Xpr[:,indx1])
    Leaves_tr2 = clf2.apply(Xtr[:,indx2])
    Leaves_pr2 = clf2.apply(Xpr[:,indx2])
    Leaves_tr = np.concatenate([Leaves_tr1, Leaves_tr2], axis=1)
    Leaves_pr = np.concatenate([Leaves_pr1, Leaves_pr2], axis=1)
    
    n_classes = 2
    n_samples = Xtr.shape[0]
    ws = n_samples / (n_classes * np.bincount(ytr))

    def DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i):
        return getTreeProbs(Leaves_tr[boots[:, i], i], ytr[boots[:, i]], ws, Leaves_pr[:, i])

    forestProbs_tr = [DebugGetTreeProbs(Leaves_tr, boots, ytr, ws, Leaves_pr, i) for i in range(Leaves_tr.shape[1])]

    probPred = predProbRF(forestProbs_tr, Leaves_pr)
    return probPred

def getIndxIncl(i, posts, C):
    indx_excl = np.arange(posts[i],posts[i+1])
    indx, = np.where(np.apply_along_axis(np.any,1,(C.to_numpy()[:,np.array(indx_excl)]>0.4)))
    indx_excl = np.sort(np.array(list(set(indx).union(indx_excl))))
    indx_incl = np.arange(posts[posts.shape[0]-1])
    indx_incl = np.array(list(set(indx_incl).difference(indx_excl)))
    return indx_incl

def getInteractionImpCluster(mods, X_train, y_train, X_val, y_val, posts, i, j, C):

    n_samples_bootstrap = X_train.shape[0]
    num_trees = 500*2
    np.random.seed(seed=12)
    smpls = np.array([np.random.randint(0, n_samples_bootstrap, n_samples_bootstrap) for i in range(num_trees)]).T

    
    mod_with = mods["mod_with"]
    mod_without = mods["mods_without"][i]
    
    indx = indx_incl = np.arange(posts[posts.shape[0]-1])
    y_pred = applyRF_comb(mods["mod_with"],mods["mod_with"], X_train, X_val, y_train, indx, indx, smpls)
    auc_with = roc_auc_score(y_val, y_pred)

    y_pred = applyRF_comb(mods["mods_without"][i],mods["mods_without"][j], X_train, X_val, y_train, getIndxIncl(i, posts, C), getIndxIncl(j, posts, C), smpls)
    auc_without = roc_auc_score(y_val, y_pred)

    imp = auc_with-auc_without
    #print("importance: ", imp)
    return imp

def getInteractionImp(mods, cube, labels, cluster, event_df, posts, C):
    indx_val, =  np.where(event_df.cluster==cluster)
    indx_train = np.array(list(set(np.arange(0, cube.shape[0])).difference(indx_val)))
    
    X_train = cube[indx_train,:]
    X_val = cube[indx_val,:]
    y_train = labels[indx_train]
    y_val = labels[indx_val]
    
    a = np.arange(posts.shape[0]-1)
    combs = list(combinations(a, 2))


    imps = [getInteractionImpCluster(mods[cluster][3][0], X_train, y_train, X_val, y_val, posts, list(combs[i])[0], list(combs[i])[1], C) for i in range(len(combs))]
    return imps







