import json
j = json.load(open("scalograms/acoustic/0Nm_BPFI_03__values__win00007.npy.meta.json"))
print(j.keys())
print(j.get("window_sec"), j.get("hop_sec"), j.get("sample_rate"))
