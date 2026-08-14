from sklearn.cluster import KMeans
import numpy as np


def make_clusters(seg_base, num_clusters: int, type:str, seed:int=42):
    """
    Make equal-sized clusters using k-means clustering.

    Args:
        seg_base (_type_): data to use for clustering
        num_clusters (int): number of clusters desired
        seed (int): random seed to  initialise model
        type (str): 'random' or 'regional'

    Returns:
        seg_base
    """
    if type == 'random':
        rnds = np.floor(np.arange(0, seg_base.shape[0])/17).astype(int)
        seg_base["cluster_random"] = np.array(rnds).astype(int)
        return seg_base
    
    if type == 'regional':
        np.random.seed(seed=seed)
        seg_base2 = seg_base[["sat", "lon", "lat"]].to_numpy()
        seg_base2 = np.array([zscore(seg_base2[:, i])
                         for i in range(seg_base2.shape[1])]).T
        N = seg_base2.shape[0]
        cluster_size = N//num_clusters
        print("cluster size: ", cluster_size)
        abs = -1*np.ones(N)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(seg_base2)
        grps = kmeans.labels_
        ids, cnt = np.unique(grps, return_counts=True)
        print(pd.DataFrame({"cluster": ids, "count": cnt}))
        o = np.argsort(cnt)
        DistMat = kmeans.fit_transform(seg_base2)

        # initialize
        indx_obs = np.arange(seg_base2.shape[0]).tolist()
        indx_cluster = ids[o][::-1].tolist()

        # loop through clusters from smallest to largest
        for i in range(len(indx_obs)):
            #print("i: ", i)
            o2 = np.argsort(DistMat[indx_obs, indx_cluster[0]])
            indx_lab = np.array(indx_obs)[o2[0]]
            labs[indx_lab] = indx_cluster[0]
            indx_cluster.pop(0)
            if len(indx_cluster) == 0:
                indx_cluster = ids[o][::-1].tolist()
            indx_obs = list(set(indx_obs).difference([indx_lab]))

        # seg_base["cluster"]=np.array(labs).astype(int)
        seg_base["cluster_regional"] = np.array(grps).astype(int)

        return seg_base


def zscore(x):
    return (x-np.mean(x))/np.std(x)
