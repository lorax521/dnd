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
from collections import OrderedDict
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

# def fight(attacker, defender):
#     attack = roll1d20()
#     if attack > defender.armorClass:
#         defender.hitPoints = defender.hitPoints - attacker.damage
#     return defender

# def battle(character, monster):
#     turn = 0
#     while (character.hitPoints > 0) & (monster.hitPoints > 0):
#         attack = roll1d20()
#         # address turn
#         if turn%2 == 0:
#             if attack > character.armorClass:
#                 character.hitPoints = character.hitPoints - monster.damage
#         if turn%2 == 1:
#             if attack > monster.armorClass:
#                 monster.hitPoints = monster.hitPoints - character.damage
#         turn += 1
#     if character.hitPoints > monster.hitPoints:
#         survival = 1
#     else:
#         survival = 0
#     return c

def createCharacters(nCharacters):
    characters = {}
    for c in range(nCharacters):
        character = classes.Character('elf', 'druid', 1)
        characters.update({str(c): character})
    return characters

def createMonsters(nMonsterRange, difficulty):
    # difficulty = genRandom(1, 5)
    nMonsters = int(genRandom(nMonsterRange[0], nMonsterRange[1]) + 0.5)
    monsters = {}
    for m in range(nMonsters):
        monster = classes.Monster(difficulty, 3)
        monsters.update({str(m): monster})
    return monsters

def initBattle(characters, monsters):
    try:
        # determine turns
        # TODO sometimes it skips a number in assigning idx
        members =  [x for x in characters.values()] + [x for x in monsters.values()]
        initiatives = [x.initiative for x in members]
        # initiatives = [8,9,16,4,6,4,5,6]
        initSorted = initiatives.copy()
        initSorted.sort()
        initNew = []
        for number in (initSorted):
            idx = initiatives.index(number)
            initNew.append(idx)
            initiatives[idx] = -1
        reMember = []
        for idx in initNew:
            reMember.append(members[idx])
        battleParty = {}
        for member in enumerate(reMember):
            battleParty.update({str(member[0]): member[1]})
        isCharacter = list(map(lambda x: isinstance(battleParty[x], classes.Character), battleParty))
        if sum([int(x) for x in battleParty]) != sum([x for x in range(0, len(battleParty))]):
            print('break')
        return battleParty, isCharacter
    except ValueError as e:
        print('---------- error ------------')
        print(e)

def determineDefender(battleParty, partyClass):
    try:
        # members of party class with hitpoints > 0
        livingParty = list(map(lambda x: (isinstance(battleParty[x], partyClass)) & (battleParty[x].hitPoints > 0), battleParty))
        # random idx of the livingParty where equals true
        idxDefender = int(genRandom(1, sum(livingParty)) + 0.5) - 1
        idxTemp = []
        idxAdditive = 0
        for m in livingParty:
            if m:
                idxTemp.append(idxAdditive)
                idxAdditive += 1
            else: 
                idxTemp.append(-1)
        defender = idxTemp.index(idxDefender)
        return defender
    except ValueError as e:
        print('-------------- error -------------')
        print(e)

def fight(attacker, defender):
    attack = roll1d20()
    if attack > defender.armorClass:
        defender.hitPoints = defender.hitPoints - attacker.damage
    return defender

def battleActive(battleParty):
    livingCharacters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty)))
    livingMonsters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Monster)) & (battleParty[x].hitPoints > 0), battleParty)))
    check = True if (livingCharacters > 0) & (livingMonsters > 0) else False
    return check

def battle(battleParty, isCharacter):
    try: 
        # df for recording results
        df = pd.DataFrame()
        # while either party still has hitpoints
        while battleActive(battleParty):
            # goes through turns by initiative
            for member in enumerate(battleParty):
                if battleActive(battleParty):
                    # checks if member is a character
                    if isCharacter[member[0]]:
                        # determines the defender
                        idxDefender = determineDefender(battleParty, classes.Monster)
                        # intitates attack and resets defender hit points
                        battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
                    else:
                        # determines the defender
                        idxDefender = determineDefender(battleParty, classes.Character)
                        # intitates attack and resets defender hit points
                        battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
        survival = int(sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty))) > 0)
        returnCharacters = {}
        charIdx = [x for x in battleParty if isinstance(battleParty[x], classes.Character)]
        [returnCharacters.update({str(x[0]): battleParty[x[1]]}) for x in enumerate(charIdx)]
        # battleCharacters = [returnCharacters.update({str(int(x) - 1): battleParty[x]}) for x in battleParty if isinstance(battleParty[x], classes.Character)]
        return survival, returnCharacters
    except ValueError as e:
        print('---------- error ------------')
        print(e)

def simulate(nCharacters, nMonsterRange, difficulty):
    try:
        wins = 0
        alive = 1
        characters = createCharacters(nCharacters)
        while alive:
            monsters = createMonsters(nMonsterRange, difficulty)
            battleParty, isCharacter = initBattle(characters, monsters)
            alive, characters = battle(battleParty, isCharacter)
            if alive:
                wins += 1
        return wins
    except ValueError as e:
        print('---------- error ------------')
        print(e)

# def simulateBattle(battles, nCharacters, nMonsterRange, plot):
#     def createCharacters(nCharacters):
#         characters = {}
#         for c in range(nCharacters):
#             character = classes.Character('elf', 'druid', 1)
#             characters.update({str(c): character})
#         return characters

#     def createMonsters(nMonsterRange, difficulty):
#         # difficulty = genRandom(1, 5)
#         nMonsters = int(genRandom(nMonsterRange[0], nMonsterRange[1]) + 0.5)
#         monsters = {}
#         for m in range(nMonsters):
#             monster = classes.Monster(difficulty, 3)
#             monsters.update({str(m): monster})
#         return monsters

#     def initBattle(characters, monsters):
#         # determine turns
#         # TODO sometimes it skips a number in assigning idx
#         members =  [x for x in characters.values()] + [x for x in monsters.values()]
#         initiatives = [x.initiative for x in members]
#         initNew = initiatives.copy()
#         for val in initiatives:
#             check = initiatives.count(val) > 1
#             if check:
#                 indicies = [x[0] for x in enumerate(initNew) if x[1] == val]
#                 increaseIdx = [x for x in range(len(initNew)) if (x not in indicies) & (initNew[x] > val)]
#                 addFactor = 0
#                 for idx in indicies:
#                     initNew[idx] = val + addFactor
#                     addFactor += 1
#                 for idx in increaseIdx:
#                     initNew[idx] = initNew[idx] + addFactor
#         initiativesSorted = sorted(initNew)
#         initiativesIdx = [initiativesSorted.index(initNew[x]) for x in range(len(initNew))]
#         membersIdx = {}
#         [membersIdx.update({str(x): members[initiativesIdx.index(x)]}) for x in initiativesIdx]
#         battleParty = OrderedDict(sorted(membersIdx.items()))
#         for i in battleParty:
#             print(battleParty[i].initiative)

#         isCharacter = list(map(lambda x: isinstance(battleParty[x], classes.Character), battleParty))
#         return battleParty, isCharacter

#     def determineDefender(battleParty, partyClass):
#         # partyClass = classes.Character if isinstance(party['0'], classes.Character) else classes.Monster
#         livingParty = list(map(lambda x: (isinstance(battleParty[x], partyClass)) & (battleParty[x].hitPoints > 0), battleParty))
#         idxDefender = int(genRandom(1, sum(livingParty)) + 0.5)
#         idxTemp = []
#         idxAdditive = 1
#         for m in livingParty:
#             if m:
#                 idxTemp.append(idxAdditive)
#                 idxAdditive += 1
#             else: 
#                 idxTemp.append(-1)
#         defender = idxTemp.index(idxDefender)
#         return defender

#     def fight(attacker, defender):
#         attack = roll1d20()
#         if attack > defender.armorClass:
#             defender.hitPoints = defender.hitPoints - attacker.damage
#         return defender

#     def battleActive(battleParty):
#         livingCharacters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty)))
#         livingMonsters = sum(list(map(lambda x: (isinstance(battleParty[x], classes.Monster)) & (battleParty[x].hitPoints > 0), battleParty)))
#         check = True if (livingCharacters > 0) & (livingMonsters > 0) else False
#         return check

#     def battle(battleParty, isCharacter):
#         # df for recording results
#         df = pd.DataFrame()
#         # while either party still has hitpoints
#         while battleActive(battleParty):
#             # goes through turns by initiative
#             for member in enumerate(battleParty):
#                 if battleActive(battleParty):
#                     # checks if member is a character
#                     if isCharacter[member[0]]:
#                         # determines the defender
#                         idxDefender = determineDefender(battleParty, classes.Monster)
#                         # intitates attack and resets defender hit points
#                         battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
#                     else:
#                         # determines the defender
#                         idxDefender = determineDefender(battleParty, classes.Character)
#                         # intitates attack and resets defender hit points
#                         battleParty[str(idxDefender)] = fight(battleParty[member[1]], battleParty[str(idxDefender)])
#         survival = int(sum(list(map(lambda x: (isinstance(battleParty[x], classes.Character)) & (battleParty[x].hitPoints > 0), battleParty))) > 0)
#         returnCharacters = {}
#         charIdx = [x for x in battleParty if isinstance(battleParty[x], classes.Character)]
#         [returnCharacters.update({str(x[0]): battleParty[x[1]]}) for x in enumerate(charIdx)]
#         # battleCharacters = [returnCharacters.update({str(int(x) - 1): battleParty[x]}) for x in battleParty if isinstance(battleParty[x], classes.Character)]
#         return survival, returnCharacters

#     # monte carlo simulation
#     df = pd.DataFrame()
#     def simulate(difficulty):
#         wins = []
#         alive = 1
#         characters = createCharacters(nCharacters)
#         while alive:
#             monsters = createMonsters(nMonsterRange, difficulty)
#             battleParty, isCharacter = initBattle(characters, monsters)
#             try:
#                 alive, characters = battle(battleParty, isCharacter)
#             except:
#                 return battleParty, isCharacter
#             print(alive)
#             wins.append(alive)
#         return wins
#     df.columns = ['difficulty', 'nCharacters', 'nMonsters', 'survival']

#     # SVM
#     y = df['survival'].values
#     # X = np.array([np.array(df['difficulty']), y]).T
#     X = np.array(df['difficulty'])
#     # Splitting the dataset into the Training set and Test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#     X_train = X_train.reshape(-1, 1)
#     X_test = X_test.reshape(-1, 1)
#     # fit classifier to the training set
#     model = SVC(gamma=2, C=1, probability=True)
#     model.fit(X_train, y_train)

#     probs = model.predict_proba(X_test)
#     preds = probs[:,1]
#     fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
#     roc_auc = metrics.auc(fpr, tpr)
#     print('ROC AUC: ' + str(roc_auc))

#     if plot:
#         visualize.plotModel(X, y, model)
    
#     return model, df, model.score(X_test, y_test)

# class BattleSuccess():
#     def __init__(self):
#         model, df, score = simulateBattle(100, 1, 1, False)
#         steps = np.linspace(1, 5, 1000)
#         self.df = pd.DataFrame()
#         for step in steps:
#             self.df = self.df.append([[step, model.predict_proba(step)[:,1][0]]])
#         self.df.columns = ['difficulty', 'probability']

#     def predictDifficulty(self, value):
#         array = np.asarray(self.df['probability'])
#         idx = (np.abs(array - value)).argmin()
#         return self.df.iloc[idx]['difficulty']

