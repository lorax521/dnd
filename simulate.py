from random import randint
from random import seed
from random import random
import classes
import pandas as pd
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
        members =  [x for x in characters.values()] + [x for x in monsters.values()]
        initiatives = [x.initiative for x in members]
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