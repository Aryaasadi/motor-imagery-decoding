from load_2a import load_subject_2a

data_dir = r"D:\Articles\1-ongoing\2-IEEE\BCICIV2a"

print("=== T session ===")
XT, yT, chT, sfT, eidT = load_subject_2a(data_dir, subj=1, session="T")
print(XT.shape, yT.shape, len(chT), sfT)
print("Counts:", {i: int((yT == i).sum()) for i in range(4)})
print("Channels:", chT)
