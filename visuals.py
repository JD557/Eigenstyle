from PIL import Image
import PIL.ImageOps
import numpy as np
from sklearn.linear_model import LogisticRegression
import random
import os
from statistics import mean, median, standard_deviation, inverse_normal_cdf, interquartile_range
import colorspaces

N_COMPONENTS = 50
N_COMPONENTS_TO_SHOW = 10
N_DRESSES_TO_SHOW = 5
N_NEW_DRESSES_TO_CREATE = 20
CONVERTER = colorspaces.BwLumaConverter()

# this is the size of all the Amazon.com images
# If you are using a different source, change the size here
STANDARD_SIZE = (200, 260)

def numComponents(pca):
    return min(N_COMPONENTS, len(pca.components_))

def img_to_array(filename):
    """takes a filename and turns it into a numpy array of RGB pixels"""
    rawImg = Image.open(filename).resize(STANDARD_SIZE)
    imgData = np.array(map(list, list(CONVERTER.apply(rawImg).getdata())))
    shape = imgData.shape[0] * imgData.shape[1]
    img_wide = imgData.reshape(1, shape)
    return img_wide[0]


def makeFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def createEigendressPictures(pca, X, raw_data):
    print("creating eigendress pictures")
    directory = "results/eigendresses/"
    makeFolder(directory)
    for i in range(min(N_COMPONENTS_TO_SHOW, numComponents(pca))):
        component = pca.components_[i]
        img = image_from_component_values(component)
        img.save(directory + str(i) + "_eigendress___.png")
        reverse_img = PIL.ImageOps.invert(img)
        reverse_img.save(directory + str(i) + "_eigendress_inverted.png")
        ranked_dresses = sorted(enumerate(X), key=lambda (a, x): x[i])
        most_i = ranked_dresses[-1][0]
        least_i = ranked_dresses[0][0]

        for j in range(N_DRESSES_TO_SHOW):
            most_j = j * -1 - 1
            Image.open(raw_data[ranked_dresses[most_j][0]][2]).save(directory + str(i) + "_eigendress__most" + str(j) + ".png")
            Image.open(raw_data[ranked_dresses[j][0]][2]).save(directory + str(i) + "_eigendress_least" + str(j) + ".png")


def indexesForImageName(imageName):
    return [i for (i, (cd, _y, f)) in enumerate(raw_data) if imageName in f]


def predictiveModeling(pca, data, X, y, raw_data):
    print("training a predictive model...")
    try:
        # split the data into a training set and a test set
        train_split = int(len(data) * 4.0 / 5.0)

        X_train = X[:train_split]
        X_test = X[train_split:]
        y_train = y[:train_split]
        y_test = y[train_split:]

        # if you wanted to use a different model, you'd specify that here
        clf = LogisticRegression(penalty='l2')
        clf.fit(X_train, y_train)

        print "score", clf.score(X_test, y_test)

        # first, let's find the model score for every dress in our dataset
        probs = zip(clf.decision_function(X), raw_data)

        prettiest_liked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == 'like' else 1, p))
        prettiest_disliked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == 'dislike' else 1, p))
        ugliest_liked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == 'like' else 1, -p))
        ugliest_disliked_things = sorted(probs, key=lambda (p, (cd, g, f)): (0 if g == 'dislike' else 1, -p))
        in_between_things = sorted(probs, key=lambda (p, (cd, g, f)): abs(p))

        # and let's look at the most and least extreme dresses
        cd = zip(X, raw_data)
        least_extreme_things = sorted(cd, key=lambda (x, (d, g, f)): sum([abs(c) for c in x]))
        most_extreme_things = sorted(cd, key=lambda (x, (d, g, f)): sum([abs(c) for c in x]), reverse=True)

        least_interesting_things = sorted(cd, key=lambda (x, (d, g, f)): max([abs(c) for c in x]))
        most_interesting_things = sorted(cd, key=lambda (x, (d, g, f)): min([abs(c) for c in x]), reverse=True)

        directory = "results/notableDresses/"
        makeFolder(directory)

        for i in range(min(N_COMPONENTS_TO_SHOW, numComponents(pca))):
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
        probs = zip(clf.decision_function(X_test), raw_data[train_split:])
        num_dislikes = len([c for c in y_test if c == 1])
        num_likes = len([c for c in y_test if c == 0])
        lowest_score = round(min([p[0] for p in probs]), 1) - 0.1
        highest_score = round(max([p[0] for p in probs]), 1) + 0.1
        INTERVAL = 0.1

        # first do the likes
        score = lowest_score
        while score <= highest_score:
            true_positives = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
            false_positives = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
            positives = true_positives + false_positives
            precision = np.float64(1.0 * true_positives) / positives
            recall = np.float64(1.0 * true_positives) / num_likes
            print "likes", score, precision, recall
            score += INTERVAL

        # then do the dislikes
        score = highest_score
        while score >= lowest_score:
            true_positives = len([p for p in probs if p[0] >= score and p[1][1] == 'dislike'])
            false_positives = len([p for p in probs if p[0] >= score and p[1][1] == 'like'])
            positives = true_positives + false_positives
            precision = np.float64(1.0 * true_positives) / positives
            recall = np.float64(1.0 * true_positives) / num_dislikes
            print "dislikes", score, precision, recall
            score -= INTERVAL

        # now do both
        score = lowest_score
        while score <= highest_score:
            likes = len([p for p in probs if p[0] <= score and p[1][1] == 'like'])
            dislikes = len([p for p in probs if p[0] <= score and p[1][1] == 'dislike'])
            print score, likes, dislikes
            score += INTERVAL
    except:
        print("the model could not be trained.")


def showHistoryOfDress(dressName, pca, raw_data, X):
    index = indexesForImageName(dressName)[0]
    directory = "results/history/dress" + str(index) + "/"
    makeFolder(directory)
    dress = X[index]
    origImage = raw_data[index][2]
    Image.open(origImage).save(directory + "dress_" + str(index) + "_original.png")
    for i in range(1, len(dress)):
        reduced = dress[:i]
        img = construct(reduced, pca)
        img.save(directory + "dress_" + str(index) + "_" + str(i) + ".png")


def bulkShowDressHistories(lo, hi, pca, X, raw_data):
    for index in range(lo, hi):
        directory = "results/history/dress" + str(index) + "/"
        makeFolder(directory)
        dress = X[index]
        origImage = raw_data[index][2]
        Image.open(origImage).save(directory + "dress_" + str(index) + "_original.png")
        for i in range(1, len(dress)):
            reduced = dress[:i]
            img = construct(reduced, pca)
            img.save(directory + "dress_" + str(index) + "_" + str(i) + ".png")


def reconstruct(dress_number, pca, X):
    eigenvalues = X[dress_number]
    img = construct(eigenvalues, pca)
    return img


def construct(eigenvalues, pca):
    components = pca.components_
    eigenzip = zip(eigenvalues, components)
    N = len(components[0])
    r = [int(sum([w * c[i] for (w, c) in eigenzip])) for i in range(N)]
    img = image_from_component_values(r)
    return img


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
    im = Image.new(CONVERTER.internalMode, STANDARD_SIZE)
    im.putdata(d)
    return CONVERTER.unapply(im)


def makeRandomDress(liked, pca, likesByComponent, dislikesByComponent):
    randomArr = []
    base = likesByComponent if liked else dislikesByComponent
    for c in base[:100]:
        mu = mean(c)
        sigma = standard_deviation(c)
        p = random.uniform(0.0, 1.0)
        num = inverse_normal_cdf(p, mu, sigma)
        randomArr.append(num)
    img = construct(randomArr, pca)
    return img


def reconstructKnownDresses(pca, X, raw_data):
    print("reconstructing dresses...")
    directory = "results/recreatedDresses/"
    makeFolder(directory)
    for i in range(N_DRESSES_TO_SHOW):
        saveName = directory + str(i)
        Image.open(raw_data[i][2]).save(saveName + "_original.png")
        img = reconstruct(i, pca, X)
        img.save(saveName + '.png')


def createNewDresses(pca, likesByComponent, dislikesByComponent):
    print("creating brand new dresses...")
    directory = "results/createdDresses/"
    makeFolder(directory)
    for i in range(N_NEW_DRESSES_TO_CREATE):
        saveNameLike = "newLikeDress" + str(i)
        saveNameDislike = "newDislikeDress" + str(i)
        randLike = makeRandomDress(True, pca, likesByComponent, dislikesByComponent)
        randLike.save('results/createdDresses/' + saveNameLike + '.png')
        randDislike = makeRandomDress(False, pca, likesByComponent, dislikesByComponent)
        randDislike.save('results/createdDresses/' + saveNameDislike + '.png')


def printComponentStatistics(pca, likesByComponent, dislikesByComponent):
    print("component statistics:\n")
    for i in range(min(N_COMPONENTS_TO_SHOW, numComponents(pca), len(likesByComponent), len(dislikesByComponent))):
        print("component " + str(i) + ":")
        likeComp = likesByComponent[i]
        dislikeComp = dislikesByComponent[i]
        print("means:                     like = " + str(mean(likeComp)) + "     dislike = " + str(mean(dislikeComp)))
        print("medians:                   like = " + str(median(likeComp)) + "     dislike = " + str(median(dislikeComp)))
        print("stdevs:                    like = " + str(standard_deviation(likeComp)) + "     dislike = " + str(standard_deviation(dislikeComp)))
        print("interquartile range:       like = " + str(interquartile_range(likeComp)) + "     dislike = " + str(interquartile_range(dislikeComp)))
        print("\n")


