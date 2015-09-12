from glob import glob
from random import shuffle
from sklearn.decomposition import RandomizedPCA
import numpy as np
import visuals

like_files = glob('images/like/*')
dislike_files = glob('images/dislike/*')
other_files = glob('images/other/*')

process_file = visuals.img_to_array

print('processing images...')
print('(this takes a long time if you have a lot of images)')
raw_data = [(process_file(filename), 'like', filename) for filename in like_files] + \
           [(process_file(filename), 'dislike', filename) for filename in dislike_files] + \
           [(process_file(filename), 'other', filename) for filename in other_files]

# randomly order the data
shuffle(raw_data)

# pull out the features and the labels
data = np.array([cd for (cd, _y, f) in raw_data])
labels = np.array([_y for (cd, _y, f) in raw_data])

print('finding principal components...')
pca = RandomizedPCA(n_components=visuals.N_COMPONENTS, random_state=0)
X = pca.fit_transform(data)
y = [1 if label == 'dislike' else 0 for label in labels]

zipped = zip(X, raw_data)
likes = [x[0] for x in zipped if x[1][1] == "like"]
dislikes = [x[0] for x in zipped if x[1][1] == "dislike"]

likesByComponent = zip(*likes)
dislikesByComponent = zip(*dislikes)
allByComponent = zip(*X)

visuals.printComponentStatistics(pca, likesByComponent, dislikesByComponent)

visuals.createEigendressPictures(pca, X, raw_data)

visuals.predictiveModeling(pca, data, X, y, raw_data)

visuals.reconstructKnownDresses(pca, X, raw_data)

visuals.bulkShowDressHistories(0, 1, pca, X, raw_data)

visuals.createNewDresses(pca, likesByComponent, dislikesByComponent)
