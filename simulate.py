from random import randint
from random import seed
from random import random
import classes
import visualize
import generic
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
seed(0)

def genRandom(min, max):
    scaledValue = min + (random() * (max - min))
    return scaledValue

def roll1d4():
    return randint(1, 4)
def roll1d6():
    return randint(1, 6)
def roll1d8():
    return randint(1, 8)
def roll1d10():
    return randint(1, 12)
def roll1d12():
    return randint(1, 12)
def roll1d20():
    return randint(1, 20)
def roll1d100():
    return randint(1, 100)

def rollInitative(listOfCreatures):
    newlistOfCreatures = []
    for creature in listOfCreatures:
        creature['rolledInitative'] = roll1d20()

def battle(character, monster):
    turn = 0
    while (character.hitPoints > 0) & (monster.hitPoints > 0):
        attack = roll1d20()
        # address turn
        if turn%2 == 0:
            if attack > character.armorClass:
                character.hitPoints = character.hitPoints - monster.damage
        if turn%2 == 1:
            if attack > monster.armorClass:
                monster.hitPoints = monster.hitPoints - character.damage
        turn += 1
    if character.hitPoints > monster.hitPoints:
        survival = 1
    else:
        survival = 0
    return character, monster, survival

def simulateBattle(battles, nCharacters, nMonsters, plot):
    # monte carlo simulation
    df = pd.DataFrame()
    for i in range(0, battles):
        difficulty = genRandom(1, 5)
        monster = classes.Monster(difficulty, 3)
        character = classes.Character('elf', 'druid', 1)
        postCharacter, postMonster, survival = battle(character, monster)
        df = df.append([[difficulty, nCharacters, nMonsters, survival]])
    df.columns = ['difficulty', 'nCharacters', 'nMonsters', 'survival']

    # SVM
    y = df['survival'].values
    # X = np.array([np.array(df['difficulty']), y]).T
    X = np.array(df['difficulty'])
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    # fit classifier to the training set
    model = SVC(gamma=2, C=1, probability=True)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC AUC: ' + str(roc_auc))

    if plot:
        visualize.plotModel(X, y, model)
    
    return model, df, model.score(X_test, y_test)

class BattleSuccess():
    def __init__(self):
        model, df, score = simulateBattle(100, 1, 1, False)
        steps = np.linspace(1, 5, 1000)
        self.df = pd.DataFrame()
        for step in steps:
            self.df = self.df.append([[step, model.predict_proba(step)[:,1][0]]])
        self.df.columns = ['difficulty', 'probability']

    def predictDifficulty(self, value):
        array = np.asarray(self.df['probability'])
        idx = (np.abs(array - value)).argmin()
        return self.df.iloc[idx]['difficulty']

