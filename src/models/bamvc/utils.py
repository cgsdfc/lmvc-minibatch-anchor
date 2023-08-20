import math


def compute_num_anchors(num_samples, num_clusters, func="log"):
    """
    func: log = nc * log2(n) ; fixed = k * nc (k=11) ;
    """
    if func == "log":
        res = math.log2(num_samples) * num_clusters
        res = int(math.ceil(res))
        if res >= num_samples:
            # 类簇数很多的极端情况，就只能取类簇数了。
            res = num_clusters
    elif func == 'fixed':
        k = 11
        while k * num_clusters >= num_samples:
            k -= 1
        res = k * num_clusters
    elif func == 'nc':
        res = num_clusters
    else:
        raise ValueError(func)

    return res
