import neuralio


def get(sample, index):
    _ , data = neuralio.eatWAV(files(sample)[index])
    return data

def getAll(sample):
    return reduce(lambda (_ ,((x, y), (xx, yy))): (x + xx, y + yy), map(neuralio.eatWAV, files(sample)), [])

def files(sample):
    if sample == "piano":
        return map(lambda x: "piano/" + str(x) + ".wav", range(1, 46))
