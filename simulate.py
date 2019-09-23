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
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
from functools import reduce
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

def fight(attacker, defender):
        attack = roll1d20()
        if attack > defender.armorClass:
            defender.hitPoints = defender.hitPoints - attacker.damage
    return defender

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

def simulateBattle(battles, nCharacters, nMonsterRange, plot):
    def createCharacters(nCharacters):
        characters = {}
        for c in range(nCharacters):
            character = classes.Character('elf', 'druid', 1)
            characters.update({str(c): character})
        return characters

    def createMonsters(nMonsterRange):
        difficulty = genRandom(1, 5)
        nMonsters = int(genRandom(nMonsterRange[0], nMonsterRange[1]) + 0.5)
        monsters = {}
        for m in range(nMonsters):
            monster = classes.Monster(difficulty, 3)
            monsters.update({str(m): monster})
        return monsters

    def initBattle(characters, monsters):
        # line 14 and 27 having problems
        # determine turns
        monsterInit = [monsters[x].initiative for x in monsters]
        charInit = [characters[x].initiative for x in characters]
        allInit = sorted(monsterInit + charInit, reverse=True)
        # order = [x[0] for x in enumerate(allInit)]
        battleParty = {}
        for stat in enumerate(allInit):
            # handles initative ties at random
            assignFactor = int(genRandom(0, 1) + 0.5)
            if assignFactor == 1:
                if stat[1] in charInit:
                    idx = charInit.index(stat[1])
                    characters[str(idx)].order = stat[0]
                    battleParty.update({str(stat[0]): characters[str(idx)]})
                else:
                    idx = monsterInit.index(stat[1])
                    monsters[str(idx)].order = stat[0]
                    battleParty.update({str(stat[0]): monsters[str(idx)]})
            else:
                if stat[1] in monsterInit:
                    idx = monsterInit.index(stat[1])
                    monsters[str(idx)].order = stat[0]
                    battleParty.update({str(stat[0]): monsters[str(idx)]})
                else:
                    idx = charInit.index(stat[1])
                    characters[str(idx)].order = stat[0]
                    battleParty.update({str(stat[0]): characters[str(idx)]})

        isCharacter = list(map(lambda x: isinstance(battleParty[x], classes.Character), battleParty))
        return battleParty, isCharacter

    def determineDefender(partyClass):
        # partyClass = classes.Character if isinstance(party['0'], classes.Character) else classes.Monster
        livingParty = list(map(lambda x: (isinstance(battleParty[x], partyClass)) & (battleParty[x].hitPoints > 0), battleParty))
        idxDefender = int(genRandom(1, sum(livingParty)) + 0.5)
        idxTemp = []
        idxAdditive = 1
        for m in livingParty:
            if m:
                idxTemp.append(idxAdditive)
                idxAdditive += 1
            else: 
                idxTemp.append(-1)
        defender = idxTemp.index(idxDefender)
        return defender

    def fight(attacker, defender):
        attack = roll1d20()
        if attack > defender.armorClass:
            defender.hitPoints = defender.hitPoints - attacker.damage
        return defender

    def checkWin(battleParty):
        livingCharacters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty)))
        livingMonsters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Monster)) & (battleParty[x].hitPoints > 0), battleParty)))
        check = True if (livingCharacters > 0) & (livingMonsters > 0) else False
        return check

    def battle(battleParty, isCharacter):
        # df for recording results
        df = pd.DataFrame()
        # while either party still has hitpoints
        # while (sum([battleParty[x].hitPoints for x in battleParty if isinstance(battleParty[x], classes.Monster)]) > 0) & (sum([battleParty[x].hitPoints for x in battleParty if isinstance(battleParty[x], classes.Character)]) > 0):
        while checkWin(battleParty):
            # goes through turns by initiative
            for member in enumerate(battleParty):
                if checkWin(battleParty):
                    # checks if member is a character
                    if isCharacter[member[0]]:
                        # determines the defender
                        idxDefender = determineDefender(classes.Monster)
                        # intitates attack and resets defender hit points
                        battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
                    else:
                        # determines the defender
                        idxDefender = determineDefender(classes.Character)
                        # intitates attack and resets defender hit points
                        battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
        survival = int(sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty))) > 0)
        returnCharacters = {}
        battleCharacters = [returnCharacters.update({str(int(x) - 1): battleParty[x]}) for x in battleParty if isinstance(battleParty[x], classes.Character)]
        print(survival)
        return survival, returnCharacters

    # monte carlo simulation
    df = pd.DataFrame()
    def simulate():
        wins = []
        alive = 1
        characters = createCharacters(nCharacters)
        while alive:
            monsters = createMonsters(nMonsterRange)
            battleParty, isCharacter = initBattle(characters, monsters)
            alive, characters = battle(battleParty, isCharacter)
            wins.append(alive)
        return wins
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

