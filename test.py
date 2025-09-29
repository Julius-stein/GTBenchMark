import torch

def compare_inmemory_dataset(ds1, ds2):
    # 直接取出内部 data 和 slices
    # 直接取出内部 data 和 slices
    data1, slices1 = ds1.data, ds1.slices
    data2, slices2 = ds2.data, ds2.slices

    print("== Keys in data ==")
    for key in set(data1.keys()) | set(data2.keys()):
        v1, v2 = getattr(data1, key, None), getattr(data2, key, None)
        if torch.is_tensor(v1) and torch.is_tensor(v2):
            same = (v1.shape == v2.shape) and torch.equal(v1, v2)
            print(f"{key:15} | shape1={tuple(v1.shape)} shape2={tuple(v2.shape)} | equal={same}")
        else:
            print(f"{key:15} | {type(v1)} vs {type(v2)}")

    print("\n== Keys in slices ==")
    for key in set(slices1.keys()) | set(slices2.keys()):
        s1, s2 = slices1.get(key), slices2.get(key)
        same = (s1.shape == s2.shape and torch.equal(s1, s2)) if (s1 is not None and s2 is not None) else False
        print(f"{key:15} | len1={len(s1) if s1 is not None else None} len2={len(s2) if s2 is not None else None} | equal={same}")


# 用法
zincA = torch.load("processed_data.pt",weights_only=False)       # 其实是 ZINC12000
zincB = torch.load("GPSprocessed_data.pt",weights_only=False)    # 另一个版本
compare_inmemory_dataset(zincA, zincB)

