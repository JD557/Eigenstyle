from PIL import Image
import PIL.ImageOps
from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import pandas as pd
import re
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression
import math
import random
import os
from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range

N_COMPONENTS = 50
N_COMPONENTS_TO_SHOW = 10
N_DRESSES_TO_SHOW = 5
N_NEW_DRESSES_TO_CREATE = 20

# this is the size of all the Amazon.com images
# If you are using a different source, change the size here
STANDARD_SIZE = (200,260)

def img_to_array(filename):
    """takes a filename and turns it into a numpy array of RGB pixels"""
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def makeFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# write out each eigendress and the dresses that most and least match it
# the file names here are chosen because of the order i wanna look at the results
# (when displayed alphabetically in finder)
def createEigendressPictures():
    print("creating eigendress pictures")
    directory = "results/eigendresses/"
    makeFolder(directory)
    for i in range(min(N_COMPONENTS_TO_SHOW, numComponents)):
        component = pca.components_[i]
        img = image_from_component_values(component)
        img.save(directory + str(i) + "_eigendress___.png")
        reverse_img = PIL.ImageOps.invert(img)
        reverse_img.save(directory + str(i) + "_eigendress_inverted.png")
        ranked_dresses = sorted(enumerate(X),
               key=lambda (a,x): x[i])
        most_i = ranked_dresses[-1][0]
        least_i = ranked_dresses[0][0]

        for j in range(N_DRESSES_TO_SHOW):
            most_j = j * -1 - 1
            Image.open(raw_data[ranked_dresses[most_j][0]][2]).save(directory + str(i) + "_eigendress__most" + str(j) + ".png")
            Image.open(raw_data[ranked_dresses[j][0]][2]).save(directory + str(i) + "_eigendress_least" + str(j) + ".png")

def indexesForImageName(imageName):
    return [i for (i,(cd,_y,f)) in enumerate(raw_data) if imageName in f]

def predictiveModeling():
    print("logistic regression...")
    directory = "results/notableDresses/"
    makeFolder(directory)

    # split the data into a training set and a test set
    train_split = int(len(data) * 4.0 / 5.0)

    X_train = X[:train_split]
    X_test = X[train_split:]
    y_train = y[:train_split]
    y_test = y[train_split:]

    # if you wanted to use a different model, you'd specify that here
    clf = LogisticRegression(penalty='l2')
    clf.fit(X_train,y_train)

    print "score",clf.score(X_test,y_test)

    # first, let's find the model score for every dress in our dataset
    probs = zip(clf.decision_function(X),raw_data)

    prettiest_liked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'like' else 1,p))
    prettiest_disliked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'dislike' else 1,p))
    ugliest_liked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'like' else 1,-p))
    ugliest_disliked_things = sorted(probs,key=lambda (p,(cd,g,f)): (0 if g == 'dislike' else 1,-p))
    in_between_things = sorted(probs,key=lambda (p,(cd,g,f)): abs(p))

    # and let's look at the most and least extreme dresses
    cd = zip(X,raw_data)
    least_extreme_things = sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]))
    most_extreme_things =  sorted(cd,key=lambda (x,(d,g,f)): sum([abs(c) for c in x]),reverse=True)

    least_interesting_things = sorted(cd,key=lambda (x,(d,g,f)): max([abs(c) for c in x]))
    most_interesting_things =  sorted(cd,key=lambda (x,(d,g,f)): min([abs(c) for c in x]),reverse=True)

    for i in range(min(N_COMPONENTS_TO_SHOW, numComponents)):
        Image.open(prettiest_liked_things[i][1][2]).save(directory + "prettiest_pretty_" + str(i) + ".png")
        Image.open(prettiest_disliked_things[i][1][2]).save(directory + "prettiest_ugly_" + str(i) + ".png")
        Image.open(ugliest_liked_things[i][1][2]).save(directory + "ugliest_pretty_" + str(i) + ".png")
        Image.open(ugliest_disliked_things[i][1][2]).save(directory + "directoryugliest_ugly_" + str(i) + ".png")
        Image.open(in_between_things[i][1][2]).save(directory + "neither_pretty_nor_ugly_" + str(i) + ".png")
        Image.open(least_extreme_things[i][1][2]).save(directory + "least_extreme_" + str(i) + ".png")
        Image.open(most_extreme_things[i][1][2]).save(directory + "most_extreme_" + str(i) + ".png")
        Image.open(least_interesting_things[i][1][2]).save(directory + "least_interesting_" + str(i) + ".png")
        Image.open(most_interesting_things[i][1][2]).save(directory + "most_interesting_" + str(i) + ".png")

    # and now let's look at precision-recall
    probs = zip(clf.decision_function(X_test),raw_data[train_split:])
    num_dislikes = len([c for c in y_test if c == 1])
    num_likes = len([c for c in y_test if c == 0])
    lowest_score = round(min([p[0] for p in probs]),1) - 0.1
    highest_score = round(max([p[0] for p in probs]),1) + 0.1
    INTERVAL = 0.1

    # first do the likes
    score = lowest_score
    while score <= highest_score:
        true_positives  = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
        false_positives = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
        positives = true_positives + false_positives
        precision = np.float64(1.0 * true_positives) / positives
        recall = np.float64(1.0 * true_positives) / num_likes
        print "likes",score,precision,recall
        score += INTERVAL

    # then do the dislikes
    score = highest_score
    while score >= lowest_score:
        true_positives  = len([p for p in probs if p[0] >= score and p[1][1] == 'dislike'])
        false_positives = len([p for p in probs if p[0] >= score and p[1][1] == 'like'])
        positives = true_positives + false_positives
        precision = np.float64(1.0 * true_positives) / positives
        recall = np.float64(1.0 * true_positives) / num_dislikes
        print "dislikes",score,precision,recall
        score -= INTERVAL

    # now do both
    score = lowest_score
    while score <= highest_score:
        likes  = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
        dislikes = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
        print score, likes, dislikes
        score += INTERVAL

def showHistoryOfDress(dressName):
    index = indexesForImageName(dressName)[0]
    directory = "results/history/dress" + str(index) + "/"
    makeFolder(directory)
    dress = X[index]
    origImage = raw_data[index][2]
    Image.open(origImage).save(directory + "dress_" + str(index) + "_original.png")
    for i in range(1,len(dress)):
        reduced = dress[:i]
        construct(reduced, directory + "dress_" + str(index) + "_" + str(i))

def bulkShowDressHistories(lo, hi):
    for index in range(lo, hi):
        directory = "results/history/dress" + str(index) + "/"
        makeFolder(directory)
        dress = X[index]
        origImage = raw_data[index][2]
        Image.open(origImage).save(directory + "dress_" + str(index) + "_original.png")
        for i in range(1,len(dress)):
            reduced = dress[:i]
            construct(reduced, directory + "dress_" + str(index) + "_" + str(i))

def reconstruct(dress_number, saveName = 'reconstruct'):
    eigenvalues = X[dress_number]
    construct(eigenvalues, saveName)

def construct(eigenvalues, saveName = 'reconstruct'):
    components = pca.components_
    eigenzip = zip(eigenvalues,components)
    N = len(components[0])
    r = [int(sum([w * c[i] for (w,c) in eigenzip]))
                     for i in range(N)]
    img = image_from_component_values(r)
    img.save(saveName + '.png')

def image_from_component_values(component):
    """takes one of the principal components and turns it into an image"""
    hi = max(component)
    lo = min(component)
    n = len(component) / 3
    divisor = hi - lo
    if divisor == 0:
        divisor = 1
    def rescale(x):
        return int(255 * (x - lo) / divisor)
    d = [(rescale(component[3 * i]),
          rescale(component[3 * i + 1]),
          rescale(component[3 * i + 2])) for i in range(n)]
    im = Image.new('RGB',STANDARD_SIZE)
    im.putdata(d)
    return im

def makeRandomDress(saveName, liked):
    randomArr = []
    base = likesByComponent if liked else dislikesByComponent
    for c in base[:100]:
        mu = mean(c)
        sigma = standard_deviation(c)
        p = random.uniform(0.0, 1.0)
        num = inverse_normal_cdf(p, mu, sigma)
        randomArr.append(num)
    construct(randomArr, 'results/createdDresses/' + saveName)

def reconstructKnownDresses():
    print("reconstructing dresses...")
    directory = "results/recreatedDresses/"
    makeFolder(directory)
    for i in range(N_DRESSES_TO_SHOW):
        Image.open(raw_data[i][2]).save(directory + str(i) + "_original.png")
        saveName = directory + str(i)
        reconstruct(i, saveName)

def createNewDresses():
    print("creating brand new dresses...")
    directory = "results/createdDresses/"
    makeFolder(directory)
    for i in range(N_NEW_DRESSES_TO_CREATE):
        saveNameLike = "newLikeDress" + str(i)
        saveNameDislike = "newDislikeDress" + str(i)
        makeRandomDress(saveNameLike, True)
        makeRandomDress(saveNameDislike, False)

def printComponentStatistics():
    print("component statistics:\n")
    for i in range(min(N_COMPONENTS_TO_SHOW, numComponents)):
        print("component " + str(i) + ":")
        likeComp = likesByComponent[i]
        dislikeComp = dislikesByComponent[i]
        print("means:                     like = " + str(mean(likeComp)) + "     dislike = " + str(mean(dislikeComp)))
        print("medians:                   like = " + str(median(likeComp)) + "     dislike = " + str(median(dislikeComp)))
        print("stdevs:                    like = " + str(standard_deviation(likeComp)) + "     dislike = " + str(standard_deviation(dislikeComp)))
        print("interquartile range:       like = " + str(interquartile_range(likeComp)) + "     dislike = " + str(interquartile_range(dislikeComp)))
        print("\n")



like_files = glob('images/like/*')
dislike_files = glob('images/dislike/*')
other_files = glob('images/other/*')

process_file = img_to_array

print('processing images...')
print('(this takes a long time if you have a lot of images)')
raw_data = [(process_file(filename),'like',filename) for filename in like_files] + \
           [(process_file(filename),'dislike',filename) for filename in dislike_files] + \
           [(process_file(filename),'other',filename) for filename in other_files]

# randomly order the data
#seed(0)
shuffle(raw_data)

# pull out the features and the labels
data = np.array([cd for (cd,_y,f) in raw_data])
labels = np.array([_y for (cd,_y,f) in raw_data])

print('finding principal components...')
pca = RandomizedPCA(n_components=N_COMPONENTS, random_state=0)
X = pca.fit_transform(data)
y = [1 if label == 'dislike' else 0 for label in labels]

zipped = zip(X, raw_data)
likes = [x[0] for x in zipped if x[1][1] == "like"]
dislikes = [x[0] for x in zipped if x[1][1] == "dislike"]

likesByComponent = zip(*likes)
dislikesByComponent = zip(*dislikes)
allByComponent = zip(*X)
numComponents = min(N_COMPONENTS, len(pca.components_))

printComponentStatistics()

createEigendressPictures()

predictiveModeling()

reconstructKnownDresses()

bulkShowDressHistories(0,1)

createNewDresses()




